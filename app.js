import transformers from "@xenova/transformers";

const { pipeline, env, AutoTokenizer, AutoModelForCausalLM } = transformers;

env.localURL = "./models";

env.remoteModels = false;

const tokenizer = await AutoTokenizer.from_pretrained("./models/onnx/quantized/cerebras/Cerebras-GPT-111M/causal-lm")
const model = await AutoModelForCausalLM.from_pretrained("./models/onnx/quantized/cerebras/Cerebras-GPT-111M/causal-lm")

// for(let i = 0; i < 5; i++){
    const userInputIds = await tokenizer.encode(">> User: Hello.",);
    const outputIds = await model.generate(userInputIds, { maxLength: 1000 });
    console.log(outputIds)
    const output = await tokenizer.decode(outputIds, true);
    console.log(output)
// }
// for step in range(5):
//     # encode the new user input, add the eos_token and return a tensor in Pytorch
//     new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

//     # append the new user input tokens to the chat history
//     bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

//     # generated a response while limiting the total chat history to 1000 tokens, 
//     chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

//     # pretty print last ouput tokens from bot
//     print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))

// const pipe = await pipeline('sentiment-analysis');

// const out = await pipe('I love transformers!');