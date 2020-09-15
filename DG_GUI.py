import pandas as pd
import re
import string
import nltk
nltk.download('wordnet')
import streamlit as st
import functools
import itertools
from nltk.stem.lancaster import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity


st.write("""
# Recommendation Engine for Customer Goal Selection


This simple app demonstrates how open text can be used to suggest the most suitable Initiatives and Assets for a Customer Goal during OSP creation. 
\nFor HCE Goal Management, suggesting the most relevent default goals will make the associated initiatives and assets accessible to the customer/ESA/CoE.

""")

stopwords = nltk.corpus.stopwords.words('english')
ps = LancasterStemmer() 


# Data import
cols = ['Default Goal ID','Default Goal Description', 'Default Goal Name', 'Linked Initiatives IDs']
data = pd.read_excel("LCX-Initiatives_into Defaul Goals-V2_AD.xlsx", usecols = cols)
data['DG_NameDesc'] = data[['Default Goal Description', 'Default Goal Name']].agg(' '.join, axis = 1)
data.head()

data2 = pd.read_excel("LCX-Initiatives_into Defaul Goals-V2_AD.xlsx", sheet_name='Initiatives')
#print(data2.columns.tolist())
data2 = data2.fillna(0)
data3 = pd.read_excel("LCX-Initiatives_into Defaul Goals-V2_AD.xlsx", sheet_name='Assets')
data3["Asset ID"] = data3["Asset ID"].str.replace(" ", "") # Remove spaces

def clean_text(txt):
    txt = "".join([c for c in txt if c not in string.punctuation]) # Discount punctuation
    tokens = re.split('\W+', txt) # Split words into a list of strings
    txt = [ps.stem(word) for word in tokens if word not in stopwords] #Stem words
    return txt


tfidf_vect = TfidfVectorizer(analyzer=clean_text)
corpus = tfidf_vect.fit_transform(data['DG_NameDesc'])



st.sidebar.header('Description of Customer Goal')

query = st.sidebar.text_input('Please enter your description here:')

st.sidebar.header('Number of Goals')
no_res = st.sidebar.slider('Number of top Default Goals to be displayed:', min_value=1, max_value=30)

# No goal types yet to lever by

def best_match(query, corpus, no_res):
    
    # Apply tf-idf and cosine similarity for query and corpus
    
    query = tfidf_vect.transform([query])
    cosineSimilarities = cosine_similarity(corpus, query, dense_output = False)
    cos_df = cosineSimilarities.toarray()
    
    
    # Generate table of of top matches
    
    Match_percent = [i*100 for i in cos_df] # calculate percentage of match 
    matches = sorted([(x,i) for (i,x) in enumerate(Match_percent)], reverse=True)[:no_res] 
    # index and percentage from cos_df
    idx = [item[1] for item in matches]
    
    matches = [item[0] for item in matches] # get the percentage
    matches = [int(float(x)) for x in matches] # convert to integer from np.array
    matches = [str(i) for i in matches] # convert int to string for percentage
    matches = list(map("{}%".format, matches))
    
    ### Must list of lists to list of integers
    
    
    Goal_Desc = [data.loc[i, 'Default Goal Description'] for i in idx] # Description of CD & KPI
    Name = [data.loc[i, 'Default Goal Name'] for i in idx]
    ID = [data.loc[i, 'Default Goal ID'] for i in idx]
    
    result = pd.DataFrame({'Goal ID': ID, 'Goal Name':Name, 'Goal Description':Goal_Desc, 'Match Percentage': matches})
    #result = pd.DataFrame(result, matches)
    
    return(result)

st.header('Top Default Goals')
output1 = best_match(query, corpus, no_res)
st.table(output1) 


def get_initiatives(*selected_indices):
    # Return the initiative for your chosen goals
    init_table = pd.DataFrame()
    
    # Must get out the Goal IDs from the original dataset as row nums are from output table
    ref = [output1['Goal ID'][i] for i in selected_indices]
    
    for i in ref:
        idx = data["Default Goal ID"][data["Default Goal ID"] == i].index.tolist()
        init = data['Linked Initiatives IDs'][idx]
    
        init = [i.replace(" ", "") for i in init] # replace space at the start of IDs
        init = [i.split(';') for i in init] 
        init = (list(itertools.chain.from_iterable(init))) # Merge list of lists
    
    
    # Refer to the second spreadsheet now
        for i in init:
            init_id = data2.loc[data2["Initiative ID"] == i]
            init_table = init_table.append(init_id)
        
    output_table=init_table[['Initiative ID', 'Initiative Name', 'Description']]

    return(output_table)

if output1.shape > (0, 0):
    selected_indices = st.multiselect('Select rows:', output1.index) 

    if selected_indices:
        output2 = get_initiatives(*selected_indices)
        st.table(output2)
    else:
        st.warning('No option is selected')
    
#output2 = get_initiatives(*selected_indices)
#st.table(output2) 


# Run with default values for both
# Check get_assets with default values and see input type, what's going wrong

def get_assets (*selected_indices2):
    # Get assets from chosen inititiatives
    
    asset_table = pd.DataFrame()
    
    # Must get out the Goal IDs from the original dataset as row nums are from output table
    #print(selected_indices2)
    ref = [output2['Initiative ID'][i] for i in selected_indices2]
    ref = list(itertools.chain.from_iterable(ref)) 
    #print(ref)
    
    for i in ref:
        #print(i)
        idx = data2["Initiative ID"][data2["Initiative ID"]== i].index.tolist()
        #idx = data2.loc[data2["Initiative ID"].isin([i]), "Initiative ID"].index.tolist()
        init = data2['Linked Assets'][idx]
        init = init.tolist()
        #print(init)
        
        if init == [0]:
            output_table = 'Assets not linked to chosen initiative'
                            
        else:                           
            init = [i.replace("\n", " ") for i in init] # replace \n space at the start of IDs
            init = [i.split(" ") for i in init] 
            init = (list(itertools.chain.from_iterable(init))) # Merge list of lists
    
    # Refer to the second spreadsheet now
            for i in init:
                asset_id = data3.loc[data3["Asset ID"] == i]
                asset_table = asset_table.append(asset_id)
        
                output_table = asset_table[['Asset ID', 'Asset Name', 'Description']]


    return(output_table)



if selected_indices != [] :
    selected_indices2 = st.multiselect('Select rows:', output2.index)

    if selected_indices2 != []:
        output3 = get_assets(selected_indices2)
        if isinstance(output3, str):
            st.text(output3)
        else:
            st.table(output3)
    else:
        st.warning('No option is selected')

#output3 = get_assets(selected_indices2)
#st.table(output3) 