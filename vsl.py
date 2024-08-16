import pandas as pd
from pyvi import ViTokenizer
from guidance import models, instruction, gen, system, user, assistant
import streamlit as st
import xml.etree.ElementTree as ET

   

rootPath = './data'
gpt = models.OpenAI("gpt-4o-mini", api_key=st.secrets.openai.api_key, echo=False)

class VSL:
  def __init__(self):
    self.original = None
    
def remove_vietnamese_accent(s):
  s = s.replace('á', 'a/')
  s = s.replace('à', 'a\\')
  s = s.replace('ả', 'a?')
  s = s.replace('ã', 'a~')
  s = s.replace('ạ', 'a.')
  s = s.replace('â', 'â')
  s = s.replace('ấ', 'â/')
  s = s.replace('ầ', 'â\\')
  s = s.replace('ẩ', 'â?')
  s = s.replace('ẫ', 'â~')
  s = s.replace('ậ', 'â.')
  s = s.replace('ă', 'ă')
  s = s.replace('ắ', 'ă/')
  s = s.replace('ằ', 'ă\\')
  s = s.replace('ẳ', 'ă?')
  s = s.replace('ẵ', 'ă~')
  s = s.replace('ặ', 'ă.')
  s = s.replace('đ', 'd')
  s = s.replace('é', 'e/')
  s = s.replace('è', 'e\\')
  s = s.replace('ẻ', 'e?')
  s = s.replace('ẽ', 'e~')
  s = s.replace('ẹ', 'e.')
  s = s.replace('ê', 'ê')
  s = s.replace('ế', 'ê/')
  s = s.replace('ề', 'ê\\')
  s = s.replace('ể', 'ê?')
  s = s.replace('ễ', 'ê~')
  s = s.replace('ệ', 'ê.')
  s = s.replace('í', 'i/')
  s = s.replace('ì', 'i\\')
  s = s.replace('ỉ', 'i?')
  s = s.replace('ĩ', 'i~')
  s = s.replace('ị', 'i.')
  s = s.replace('ó', 'o/')
  s = s.replace('ò', 'o\\')
  s = s.replace('ỏ', 'o?')
  s = s.replace('õ', 'o~')
  s = s.replace('ọ', 'o.')
  s = s.replace('ô', 'ô')
  s = s.replace('ố', 'ô/')
  s = s.replace('ồ', 'ô\\')
  s = s.replace('ổ', 'ô?')
  s = s.replace('ỗ', 'ô~')
  s = s.replace('ộ', 'ô.')
  s = s.replace('ơ', 'ơ')
  s = s.replace('ớ', 'ơ/')
  s = s.replace('ờ', 'ơ\\')
  s = s.replace('ở', 'ơ?')
  s = s.replace('ỡ', 'ơ~')
  s = s.replace('ợ', 'ơ.')
  s = s.replace('ú', 'u/')
  s = s.replace('ù', 'u\\')
  s = s.replace('ủ', 'u?')
  s = s.replace('ũ', 'u~')
  s = s.replace('ụ', 'u.')
  s = s.replace('ư', 'ư')
  s = s.replace('ứ', 'ư/')
  s = s.replace('ừ', 'ư\\')
  s = s.replace('ử', 'ư?')
  s = s.replace('ữ', 'ư~')
  s = s.replace('ự', 'ư.')
  s = s.replace('ý', 'y/')
  s = s.replace('ỳ', 'y\\')
  s = s.replace('ỷ', 'y?')
  s = s.replace('ỹ', 'y~')
  s = s.replace('ỵ', 'y.')
  return s  
  
def preprocess(s):
  # define a mapping table, and use the translate function to replace the characters
  # thứ hai, thứ ba, thứ tư, thứ năm, thứ sáu, thứ bảy -> thứ_2, thứ_3, thứ_4, thứ_5, thứ_6, thứ_7
  
  #mapping
  mapping = {
    'thứ hai': 'thứ_2',
    'thứ ba': 'thứ_3',
    'thứ tư': 'thứ_4',
    'thứ_tư': 'thứ_4', 
    'thứ năm': 'thứ_5',
    'thứ sáu': 'thứ_6',
    'thứ bảy': 'thứ_7'
  }
  
  # replace using str.makestrans
  for key, value in mapping.items():
    s = s.replace(key, value)
    
  return s  
  

def llm_qa_train (lm, system_prompt, samples):
  with system():
    lm += system_prompt

  for sample in samples:
    with user():
      lm += sample[0]
    with assistant():
      lm += sample[1]

  return lm

def llm_qa_generator (lm, q, name):
  with user():
    lm += q
  with assistant():
    lm += f'''{gen(stop='', name=name, list_append=True)}'''

  return lm

def train():
  seed = 28290

  system_prompt = "You are a sign language translation machine. Please rephrase the given Vietnamese sentences (S-V-O structure, where S is the subject, V is the verb, and O is the complement) using Vietnamese Sign Language grammar (basically in S-O-V structure), with a focus on simplifying and emphasizing key elements. Just return new sentence. No yapping."

  # open vsl_train.csv set, get random n rows, and convert inito sample_pairs [['', '']]
  n = 20
  df = pd.read_csv(f'{rootPath}/vsl_train.csv', header=0)
  df = df.sample(n, random_state=seed)
  df = df.reset_index(drop=True)

  sample_pairs = [[row['Input'], row['Output']] for index, row in df.iterrows()]
  
  lm = gpt
  lm = llm_qa_train(lm, system_prompt, sample_pairs)

  return lm


 
   
lm_vsl = train()

input =  'synonyms.csv'      
synonyms_df = pd.read_csv(f'{rootPath}/{input}', header=0)
synonyms_df = synonyms_df.fillna('')

input = 'vsldict.csv'
vsldict = pd.read_csv(f'{rootPath}/{input}', header=0)     
synonyms_df = synonyms_df.fillna('')


def convertToSiGML(str):
  # tokenize 

  # str = preprocess(str)
  tokenize = ViTokenizer.tokenize(str).lower()
  tokenize = preprocess(tokenize)
  # convert to vsl

  lm_vsl_inference = lm_vsl

  vsl = llm_qa_generator(lm_vsl_inference, tokenize, 'vsl_result')
  vsl = vsl['vsl_result'][-1].strip().translate(str.maketrans('', '', ',.?!'))


  words = vsl.split()

  # convert words to dataframe

  words_df = pd.DataFrame(words, columns=['Word'])

  # find word in vsldict, return "Hand" column

  words_df['Hand'] = words_df['Word'].apply(lambda x: vsldict.loc[vsldict['Word'] == x, 'Hand'].values[0] if len(vsldict.loc[vsldict['Word'] == x, 'Hand'].values) > 0 else None)

  # if words_df['Hand'] = None, check in synonyms, add synonym column,

  synonyms = [None] * words_df.shape[0]
  for index, row in words_df.iterrows():
    word = row['Word']
    if row['Hand'] is None:
      if word in synonyms_df['Word'].values:
        synonyms[index] = synonyms_df.loc[synonyms_df['Word'] == word, 'Semilar'].values[0]

      elif word in synonyms_df['Semilar'].values:
        synonyms[index] = synonyms_df.loc[synonyms_df['Semilar'] == word, 'Word'].values[0]

  words_df['Synonyms'] = synonyms

  # find words_df['Synonyms'] again in vsldict, return Hand

  for index, row in words_df.iterrows():
    word = row['Synonyms']
    if row['Hand'] is None:
      if word is not None and word in vsldict["Word"].values:
        words_df.at[index, 'Hand'] = vsldict.loc[vsldict['Word'] == word, 'Hand'].values[0]

  # find words_df['Hand'] = None, and split 'Word' into characters, insert into df in 'Final' column, if ['Hand'] not None, keep Final = 'Word'

  for index, row in words_df.iterrows():
    if row['Hand'] is None:
      # split into charactor remove _, remove_vietnamese_accent
      
      word = remove_vietnamese_accent(row['Word'])
      
      words_df.at[index, 'Final'] = ' '.join([char for char in word if char != '_'])
    else:
      words_df.at[index, 'Final'] = row['Word']

  # create new df with 'Final' column split by ' ', keep 'Hand', 'Synonyms' if not split
  words_final_df = words_df['Final'].str.split(expand=True).stack().reset_index(level=1, drop=True).to_frame('Final')
  words_final_df = words_final_df.join(words_df[['Hand', 'Synonyms']])
  words_final_df = words_final_df.reset_index(drop=True)

  # find in vsldict again when 'Hand' is None

  for index, row in words_final_df.iterrows():
    if row['Hand'] is None:
      if row['Final'] in vsldict["Word"].values:
        result = vsldict.loc[vsldict['Word'] == row['Final'], 'Hand'].values
        words_final_df.at[index, 'Hand'] = result
        # make sure it string, not array
        if len(result) > 0:
          words_final_df.at[index, 'Hand'] = result[0]



  # clear 'Hand', strip, keep character, number, and comma only

  # words_final_df['Hand'] = words_final_df['Hand'].apply(lambda x: x.strip() if x is not None else None)
  # words_final_df['Hand'] = words_final_df['Hand'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)) if x is not None else None)

  return [tokenize, vsl, words_final_df]

def dictToDataFrame(words ):
  #match words with vsldict
  words_df = pd.DataFrame(words, columns=['Word'])
  words_df['Final'] = words_df['Word']
  words_df['Hand'] = words_df['Word'].apply(lambda x: vsldict.loc[vsldict['Word'] == x, 'Hand'].values[0] if len(vsldict.loc[vsldict['Word'] == x, 'Hand'].values) > 0 else None)
  return words_df  

def convertToSigMLXML(sigml_df):
  # Create the root element
  root = ET.Element('sigml')

  # Iterate over the DataFrame rows
  for _, row in sigml_df.iterrows():
    word = row["Final"]
    hand = row["Hand"] if row["Hand"] is not None else ''

    # Create a 'hamnosys' element for each row
    hns_sign = ET.SubElement(root, 'hns_sign')
    hns_sign.set('gloss', word)
    hamnosys_manual = ET.SubElement(hns_sign, 'hamnosys_manual')

    # splits hand and generate each element in hamnosys_manual
    for h in hand.split(','):
      ET.SubElement(hamnosys_manual, h)

  # Convert the XML tree to a string
  xml_string = ET.tostring(root, encoding='unicode')

  return xml_string
