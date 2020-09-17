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


#st.write("""
# Create New Goal


#This simple app demonstrates how open text can be used to suggest the most suitable Initiatives and Assets for a Customer Goal during OSP creation. 
#\nFor HCE Goal Management, suggesting the most relevent default goals will make the associated initiatives and assets accessible to the customer/ESA/CoE.

#""")



stopwords = nltk.corpus.stopwords.words('english')
ps = LancasterStemmer() 


# Data import
cols = ['Default Goal ID','Default Goal Description', 'Default Goal Name', 'Linked Initiatives IDs']
data = pd.read_excel("LCX-Initiatives_into Defaul Goals-V2_AD.xlsx", usecols = cols)
data['DG_NameDesc'] = data[['Default Goal Description', 'Default Goal Name']].agg(' '.join, axis = 1)
#data.head()

data2 = pd.read_excel("LCX-Initiatives_into Defaul Goals-V2_AD.xlsx", sheet_name='Initiatives')
data2['I_NameDesc'] = data2[['Description', 'Initiative Name']].agg(' '.join, axis = 1)
data2 = data2.fillna(0)

data3 = pd.read_excel("LCX-Initiatives_into Defaul Goals-V2_AD.xlsx", sheet_name='Assets')
data3["Asset ID"] = data3["Asset ID"].str.replace(" ", "") # Remove spaces
data3["Asset Name"] = data3["Asset Name"].astype(str)
data3["Description"] = data3["Description"].astype(str)
data3['A_NameDesc'] = data3[['Description', 'Asset Name']].agg(' '.join, axis = 1)




def clean_text(txt):
    txt = "".join([c for c in txt if c not in string.punctuation]) # Discount punctuation
    tokens = re.split('\W+', txt) # Split words into a list of strings
    txt = [ps.stem(word) for word in tokens if word not in stopwords] #Stem words
    return txt


tfidf_vect = TfidfVectorizer(analyzer=clean_text)
corpus = tfidf_vect.fit_transform(data['DG_NameDesc'])



st.title('Customer Goal Name')
query = st.text_input('Please enter your description here:')


#st.sidebar.header('Number of Goals')
#no_res = st.sidebar.slider('Number of top Default Goals to be displayed:', min_value=1, max_value=30)


################## Model Matches ######################

def best_match(query, corpus):
    
    # Apply tf-idf and cosine similarity for query and corpus
    
    query = tfidf_vect.transform([query])
    cosineSimilarities = cosine_similarity(corpus, query, dense_output = False)
    cos_df = cosineSimilarities.toarray()
    
    
    # Generate table of of top matches
    
    Match_percent = [i*100 for i in cos_df] # calculate percentage of match 
    matches = sorted([(x,i) for (i,x) in enumerate(Match_percent)], reverse=True)
    # index and percentage from cos_df
    idx = [item[1] for item in matches]
    
    matches = [item[0] for item in matches] # get the percentage
    matches = [int(float(x)) for x in matches] # convert to integer from np.array
    matches =  [i for i in matches if i >= 20] # remove those lower than 20%
    matches = [str(i) for i in matches] # convert int to string for percentage
    matches = list(map("{}%".format, matches))

    # take first n elements of idx to mirror matches
    idx = idx[:len(matches)]
    
    ### Must list of lists to list of integers
    
    
    Goal_Desc = [data.loc[i, 'Default Goal Description'] for i in idx] # Description of CD & KPI
    Name = [data.loc[i, 'Default Goal Name'] for i in idx]
    ID = [data.loc[i, 'Default Goal ID'] for i in idx]
    
    result = pd.DataFrame({'Goal ID': ID, 'Goal Name':Name, 'Goal Description':Goal_Desc, 'Match Percentage': matches})
    #result = pd.DataFrame(result, matches)
    
    return(result)

if query != "":
    st.header('Top Default Goals')
    output1 = best_match(query, corpus)
    st.table(output1) 

init_corpus = tfidf_vect.fit_transform(data2['I_NameDesc'])

def best_match_init(query, corpus):
    
    # Apply tf-idf and cosine similarity for query and corpus
    
    query = tfidf_vect.transform([query])
    cosineSimilarities = cosine_similarity(corpus, query, dense_output = False)
    cos_df = cosineSimilarities.toarray()
    
    
    # Generate table of of top matches
    
    Match_percent = [i*100 for i in cos_df] # calculate percentage of match 
    matches = sorted([(x,i) for (i,x) in enumerate(Match_percent)], reverse=True)
    # index and percentage from cos_df
    idx = [item[1] for item in matches]
    
    matches = [item[0] for item in matches] # get the percentage
    matches = [int(float(x)) for x in matches] # convert to integer from np.array
    matches =  [i for i in matches if i >= 20] # remove those lower than 20%
    matches = [str(i) for i in matches] # convert int to string for percentage
    matches = list(map("{}%".format, matches))
    
    
    # take first n elements of idx to mirror matches
    idx = idx[:len(matches)]
    

    ### Must list of lists to list of integers
    
    
    Init_Desc = [data2.loc[i, 'Description'] for i in idx] # Description of CD & KPI
    Name = [data2.loc[i, 'Initiative Name'] for i in idx]
    ID = [data2.loc[i, 'Initiative ID'] for i in idx]
    
    result = pd.DataFrame({'Initiative ID': ID, 'Initiative Name':Name, 'Description':Init_Desc, 'Match Percentage': matches})
    #result = pd.DataFrame(result, matches)
    
    return(result)


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



if query != "" and output1.shape > (0, 0):
    selected_indices = st.multiselect('Select rows:', output1.index) 

    if selected_indices:
        st.header("Linked Initiatives:")
        output2 = get_initiatives(*selected_indices)
        
        # Best match intiatives
        output2a = best_match_init(query, init_corpus)


        output2b = output2a[~output2a["Initiative ID"].isin(output2["Initiative ID"])] # Values in match not in linked results
        st.table(output2)

        if output2b.empty:
            st.header("Recommendation Engine results match associated initiatives as per the Default Goal Catalogue.")
        else:
            st.header("Recommended Engine suggestions not linked in the Default Goal Catalogue:")
            st.table(output2b)

    else:
        st.warning('No option is selected')
    


asset_corpus = tfidf_vect.fit_transform(data3['A_NameDesc'])

def best_match_asset(query, corpus):
    
    # Apply tf-idf and cosine similarity for query and corpus
    
    query = tfidf_vect.transform([query])
    cosineSimilarities = cosine_similarity(corpus, query, dense_output = False)
    cos_df = cosineSimilarities.toarray()
    
    
    # Generate table of of top matches
    
    Match_percent = [i*100 for i in cos_df] # calculate percentage of match 
    matches = sorted([(x,i) for (i,x) in enumerate(Match_percent)], reverse=True)
    # index and percentage from cos_df
    idx = [item[1] for item in matches]
    
    matches = [item[0] for item in matches] # get the percentage
    matches = [int(float(x)) for x in matches] # convert to integer from np.array
    matches =  [i for i in matches if i >= 20] # remove those lower than 20%
    matches = [str(i) for i in matches] # convert int to string for percentage
    matches = list(map("{}%".format, matches))
    
    # take first n elements of idx to mirror matches
    idx = idx[:len(matches)]

    ### Must list of lists to list of integers
    
    
    A_Desc = [data3.loc[i, 'Description'] for i in idx] # Description of CD & KPI
    Name = [data3.loc[i, 'Asset Name'] for i in idx]
    ID = [data3.loc[i, 'Asset ID'] for i in idx]
    
    result = pd.DataFrame({'Asset ID': ID, 'Asset Name':Name, 'Description':A_Desc, 'Match Percentage': matches})

    
    return(result)

def get_assets (*selected_indices2):
    # Get assets from chosen inititiatives
    
    asset_table = pd.DataFrame()
    
    # Must get out the Goal IDs from the original dataset as row nums are from output table
    #print(selected_indices2)
    ref = [output2['Initiative ID'][i] for i in selected_indices2]
    ref = list(itertools.chain.from_iterable(ref)) 

    for i in ref:
        idx = data2["Initiative ID"][data2["Initiative ID"]== i].index.tolist()
        init = data2['Linked Assets'][idx]
        init = init.tolist()
        
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



if query != "" and selected_indices != [] :
    
    selected_indices2 = st.multiselect('Select rows:', output2.index)
    
    if selected_indices2 != []:
        st.header("Linked Assets:")
        output3 = get_assets(selected_indices2)
        
        if isinstance(output3, str):
            st.text(output3)
        else:
            st.table(output3)

            # Best match assets
            output3a = best_match_asset(query, asset_corpus)
        
        
            output3b = output3a[~output3a["Asset ID"].isin(output3["Asset ID"])] # Values in match not in linked results
        

            if output3b.empty:
                st.header("Model results match associated assets as per the Default Goal Catalogue.")
            else:
                st.header("Recommended Engine suggestions not linked in the Default Goal Catalogue:")
                st.table(output3b)
        
    else:
        st.warning('No option is selected')

#st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)