import time
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
import openai
import pandas as pd
import numpy as np
import tiktoken 

TARGET_URL = "https://www.taiwanmobile.com/cs/public/faq/queryList.htm"
GPT_MODEL = "gpt-3.5-turbo"  # only matters insofar as it selects which tokenizer to use
EMBEDDING_MODEL = "text-embedding-ada-002"

driver = webdriver.Chrome()
driver.get(TARGET_URL)

select_element = driver.find_element(By.NAME, 'svcType')
select = Select(select_element)
select_options_value_dict = {}
for _ in range(len(select.options)):
    select.select_by_index(_)
    time.sleep(0.5)
    opt_list = []
    for opt in driver.find_element(By.NAME, 'productType').find_elements(By.TAG_NAME, "option"):
        opt_list.append(opt.get_attribute("value"))
    name = select.options[_].get_property('value')
    select_options_value_dict[name] = opt_list
    time.sleep(1)

driver.quit()

def parse_elemnet_qa(soup):
    qa_list = []
    svcType = soup.find(id='svcType').find('option',selected=True).text.strip()
    productType = soup.find(id='productType').find('option',selected=True).text.strip()
    for bq in soup.find_all('blockquote', class_='v2-page-pay__faq-column'):
        if bq.find(class_='v2-uikit__typography-text'):
            question = bq.find(class_='v2-uikit__typography-text').text.strip()
            answer = bq.find(class_='v2-m-faq-card__description').text.strip()
            qa_list.append({'svcType': svcType, 'productType': productType, 'question': question, 'answer': answer})
    return qa_list

total_qa_list = []
for svc, prod_list in select_options_value_dict.items():
    for prod in prod_list:
        resp = requests.get(TARGET_URL, params={'svcType': svc, 'productType': prod})
        soup = BeautifulSoup(resp.text)
        total_qa_list.append(parse_elemnet_qa(soup))
        time.sleep(1)

cmp_list = []
for qas in total_qa_list:
    for qa in qas:
        cmp_list.append(qa)

df = pd.DataFrame(cmp_list)
df.to_csv("data/twm_common_questions.csv", index=False)

def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    """Return the number of tokens in a string."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

total_token_nums = []
for qa in cmp_list:
    text = qa['question'] + '\n' + qa['answer']
    total_token_nums.append(num_tokens(text))
print("max number of tokens: ", np.max(total_token_nums))

openai_api_key = input("please enter your openai api key: ")
openai.api_key = openai_api_key

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']

qa_merged_list = [qa['question'] + ' ' + qa['answer'] for qa in cmp_list]
text = qa_merged_list[0].replace('\n', ' ')
embed = openai.Embedding.create(input = [text], model=EMBEDDING_MODEL)['data'][0]['embedding']

df['text'] = df['question'] + ' ' + df['answer']
df['ada_embedding'] = df.text.apply(lambda x: get_embedding(x, model=EMBEDDING_MODEL))
df[['text','ada_embedding']].to_csv('data/twm_text_embedded.csv', index=False)