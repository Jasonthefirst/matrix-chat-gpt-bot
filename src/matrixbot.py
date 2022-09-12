from getpass import GetPassWarning
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
from pynvml import *

import simplematrixbotlib as botlib

import json

class NLPResponder:
    
    
    def __init__(self, model_name, use_cuda):
        
        POS_TOKEN = "<|review_pos|>"
        NEG_TOKEN = "<|review_neg|>"
        BOS_TOKENS = [NEG_TOKEN, POS_TOKEN]
        self.EOS_TOKEN = "<|endoftext|>"
        self.PAD_TOKEN = "<|pad|>"
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        #current_app.model = AutoModelForCausalLM.from_pretrained(model_name, revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True)

        free_vram = 0.0
        if torch.cuda.is_available():
            
            nvmlInit()
            h = nvmlDeviceGetHandleByIndex(0)
            info = nvmlDeviceGetMemoryInfo(h)
            free_vram = info.free/1048576000
            print("There is a GPU with " + str(free_vram) + "GB of free VRAM")
        # right now: use always cuda (GPU). If you don't want to use the gpu, comment this out. 
        if model_name == "EleutherAI/gpt-neo-125M" and free_vram>2.5:
            self.use_cuda = True
            self.model.to("cuda:0")
        elif model_name == "EleutherAI/gpt-neo-2.7B" and free_vram>13.5:
            self.use_cuda = True
            self.model.to("cuda:0")
        elif model_name == "EleutherAI/gpt-neo-1.3B" and free_vram>7.5:
            self.use_cuda = True
            self.model.to("cuda:0")
        elif model_name == "EleutherAI/gpt-j-6B" and free_vram >20:
            self.use_cuda = True
            self.model.to("cuda:0")
        else:
            self.use_cuda = use_cuda
            if self.use_cuda:
                self.model.to("cuda:0")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, eos_token=self.EOS_TOKEN, pad_token=self.PAD_TOKEN)

        print("model loaded in GPU: " + str(use_cuda))  
        
        self.conversationString = ""
        self.lastconversationString = ""
        self.prompt_temperature = 0.3
        self.prompt_length = 60
        self.freq_pen = 30.0
        self.top_p = 1
        self.request_count = 0
            
    def request_prompt(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
                
        if self.use_cuda:
            input_ids = input_ids.cuda()
            print("using cuda")
        else:
            print("not using cuda")

        output_prompt_length = input_ids.size(1) + self.prompt_length

        
        gen_tokens = self.model.generate(input_ids, eos_token=self.EOS_TOKEN, do_sample=True, top_p=self.top_p, repetition_penalty=self.freq_pen, temperature=self.prompt_temperature, max_length=output_prompt_length)
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]

        return gen_text
    
    def clearConversationstring(self):
        self.lastconversationString =""
        self.request_count = 0

#the model to use. the smaller the model the faster it works (and you don't need a big gpu) but the text might not be as good.
gpttransformer = NLPResponder("EleutherAI/gpt-neo-125M", True)   
#gpttransformer = NLPResponder("EleutherAI/gpt-neo-1.3B", True)
#gpttransformer = NLPResponder("facebook/opt-13b", True)
#gpttransformer = NLPResponder("seyonec/ChemBERTa-zinc-base-v1", True)
#gpttransformer = NLPResponder("EleutherAI/gpt-j-6B", True)
#gpttransformer = NLPResponder("bigscience/bloom-7b1", True)
#gpttransformer = NLPResponder("facebook/opt-6.7b", True)
#gpttransformer = NLPResponder("facebook/opt-125m", True)

#create a bot user and password in your matrix instance for this to work
creds = botlib.Creds("https://matrix.YOURINSTANCE.com", "bot-gptbot", "YOURGPTPASSWORD")
bot = botlib.Bot(creds)
PREFIX = '!'

@bot.listener.on_message_event
async def bot_help(room, message):
    bot_help_message = f"""
    Help Message:
        prefix: {PREFIX}
        commands:
            help:
                command: help, ?, h
                description: display help command
            clear:
                command: clear
                description: clears the response!
            settemp:
                command: !settemp
            setlength:
                command: !settlength
            settopp:
                command: !settopp
            setfreqpen
                command: !setfreqpen
                description: sets the penalty for repeating itself
                """
    match = botlib.MessageMatch(room, message, bot, PREFIX)
    if match.is_not_from_this_bot() and match.prefix() and (
            match.command("help") or match.command("?") or match.command("h")):
        await bot.api.send_text_message(room.room_id, bot_help_message)

@bot.listener.on_message_event
async def clear(room, message):
    match = botlib.MessageMatch(room, message, bot, PREFIX)

    if match.is_not_from_this_bot() and match.prefix() and match.command("clear"):
        gpttransformer.clearConversationstring()
        await bot.api.send_text_message(
            room.room_id, "Cleared everything!"
            )

@bot.listener.on_message_event
async def echo(room, message):
    match = botlib.MessageMatch(room, message, bot, PREFIX)

    if match.is_not_from_this_bot() and match.prefix() and match.command("echo"):

        await bot.api.send_text_message(
            room.room_id, " ".join(arg for arg in match.args())
            )

@bot.listener.on_message_event
async def setTemp(room, message):
    match = botlib.MessageMatch(room, message, bot, PREFIX)

    if match.is_not_from_this_bot() and match.prefix() and match.command("settemp"):
        gpttransformer.prompt_temperature = float(match.args()[0])
        await bot.api.send_text_message(
            room.room_id, "Temp set to: ".join(str(gpttransformer.prompt_temperature))
            )
        
@bot.listener.on_message_event
async def setFreqPenalty(room, message):
    match = botlib.MessageMatch(room, message, bot, PREFIX)

    if match.is_not_from_this_bot() and match.prefix() and match.command("setfreqpen"):
        gpttransformer.freq_pen = float(match.args()[0])
        await bot.api.send_text_message(
            room.room_id, "Frequency penalty set to: ".join(str(gpttransformer.freq_pen))
            )
        
@bot.listener.on_message_event
async def settopP(room, message):
    match = botlib.MessageMatch(room, message, bot, PREFIX)

    if match.is_not_from_this_bot() and match.prefix() and match.command("settopp"):
        gpttransformer.top_p = float(match.args()[0])
        await bot.api.send_text_message(
            room.room_id, "Top P set to: ".join(str(gpttransformer.top_p))
            )

@bot.listener.on_message_event
async def setLength(room, message):
    match = botlib.MessageMatch(room, message, bot, PREFIX)

    if match.is_not_from_this_bot() and match.prefix() and match.command("setlength"):
        gpttransformer.prompt_length = int(match.args()[0])
        await bot.api.send_text_message(
            room.room_id, "Length set to: " + (str(gpttransformer.prompt_length))
            )

@bot.listener.on_message_event
async def respondwithGPT(room, message):
    
    match = botlib.MessageMatch(room, message, bot, PREFIX)

    if match.is_not_from_this_bot() and not match.prefix():
        
        convString = gpttransformer.lastconversationString
        
        #message = str(message).split(':', 2)[1:]
        message_content = ''.join((str(message).split(':')[2:]))
        
        if len(gpttransformer.lastconversationString) < 2:
            prompt_engineering = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\nHuman: Hello, who are you?\nAI: I am an AI. How can I help you today?\n"
            
            message_content = prompt_engineering + "Human: " + message_content + "\nAI: "
        else:
            message_content = "\nHuman: " + message_content + "\nAI: "
            pass
            
        convString = convString + str(message_content)
    
        gpttransformer.lastconversationString = convString
    
        answer = gpttransformer.request_prompt(convString)
        print("answer from GPT before splits: ", answer)
        answer = answer.replace("</s>", '')
        
        answer = answer.replace(convString, '')
        
        answer = ''.join(answer.split('AI:', 3 + gpttransformer.request_count)[:2])
        answer = ''.join(answer.split('Human:', 3 + gpttransformer.request_count)[:2])
        answer = ''.join(answer.split('A:', 1)[:2])
        print("answer from GPT: ", answer)
        gpttransformer.request_count += 1 
        gpttransformer.lastconversationString = gpttransformer.lastconversationString + answer
                
        await bot.api.send_text_message(
            room.room_id,
            answer)
            


bot.run()