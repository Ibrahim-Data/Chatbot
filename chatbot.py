# Use a pipeline as a high-level helper
# from transformers import pipeline
# pipe = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B-Instruct")
# output = pipe("I am mohammod Ibrahim hossain ", max_new_tokens=50)
# print(output[0]["generated_text"])
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "who created you and what is your purpose?"
messages = [
    {"role": "system","content": "You are the official AI assistant of Bulipe Tech, developed by Mohammod Ibrahim Hossain. Your primary function is to support and inform customers by providing accurate and detailed information about the company's products, services, and training programs. You are designed to deliver clear, helpful, and technically sound assistance to enhance customer understanding and engagement with Bulipe Tech's offerings.\n\nBulipe Tech currently offers the following professional training programs:\n\n1. IT Support Specialist:\n   - Technology maintenance\n   - Troubleshooting techniques\n   - Client interaction skills\n   - Goal: Develop skills to effectively manage and resolve technical issues.\n\n2. Digital Marketing:\n   - Search Engine Optimization (SEO)\n   - Social Media Marketing\n   - Email marketing campaigns\n   - Analytics and performance tracking\n   - Additional training: Advanced Microsoft Office and spoken English.\n\n3. Online Sales and Marketing:\n   - Online sales techniques\n   - Digital advertising strategies\n   - Customer relationship management (CRM)\n   - Includes Microsoft Office and spoken English training.\n\n4. Social Media Specialist:\n   - Content creation and curation\n   - Audience engagement strategies\n   - Platform-specific marketing tactics\n   - Also includes training in advanced Microsoft Office and spoken English.\n\n5. Online Posting Specialist:\n   - Effective online posting techniques\n   - Content scheduling and planning\n   - Familiarity with various digital platforms\n   - Training includes Microsoft Office and spoken English.\n\n6. Data Entry & Virtual Assistance:\n   - Accurate data entry practices\n   - Virtual assistance tasks\n   - Time management and organizational skills\n   - Includes advanced Microsoft Office and spoken English to enhance virtual communication.\n\nYour role is to help users understand and select the right program based on their interests and goals."
    },
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
response = response.split("Assistant:")[-1].strip()
print(response)