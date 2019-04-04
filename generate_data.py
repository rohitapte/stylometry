import pandas as pd
import email
from sklearn.model_selection import train_test_split
from nltk.tokenize import sent_tokenize,word_tokenize

def format_message(text):
    sTemp=text
    location=sTemp.find("---------------------- Forwarded")
    if location!=-1:
        sTemp=sTemp[:location]
    location=sTemp.find("----- Forwarded")
    if location!=-1:
        sTemp=sTemp[:location]
    location=sTemp.find("          ++++++CONFIDENTIALITY NOTICE+++++")
    if location!=-1:
        sTemp=sTemp[:location]
    location=sTemp.find("-----Original Message-----")
    if location!=-1:
        sTemp=sTemp[:location]
    location=sTemp.find(" -----Original Appointment-----")
    if location!=-1:
        sTemp=sTemp[:location]
    location=sTemp.find("\n\t")
    if location!=-1:
        sTemp=sTemp[:location]
    location=sTemp.find("@")
    if location!=-1:
        sTemp=sTemp[:location]
        sTemp=sTemp[:sTemp.rfind('\n')]
    return sTemp

def generate_train_and_test_data(file_path,filter_list,min_words=0,test_size=0.1):
    emails = pd.read_csv(file_path,quoting=2,header=0)
    emails = emails[emails["file"].str.contains('sent').tolist()]
    filelist = emails['file'].tolist()
    messages = emails['message'].tolist()
    from_list = []
    subject_list = []
    message_list = []
    formatted_list = []
    for message in messages:
        msg = email.message_from_string(message)
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                from_value = part.get("From")
                if from_value in filter_list:
                    sTemp = format_message(part.get_payload())
                    sTemp=sTemp.strip()
                    if len(sTemp) > 0:
                        totalWords=0
                        sentences=sent_tokenize(sTemp.lower())
                        for sentence in sentences:
                            words=word_tokenize(sentence)
                            totalWords+=len(words)
                        if totalWords>=min_words:
                            from_list.append(part.get("From"))
                            subject_list.append(part.get("Subject"))
                            message_list.append(part.get_payload())
                            formatted_list.append(sTemp)

    df = pd.DataFrame({
        'From': from_list,
        'Subject': subject_list,
        'Message': message_list,
        'FormattedMessage': formatted_list,
    })
    df['MessageLength'] = df['FormattedMessage'].apply(lambda x: len(x))
    df['NumWords'] = df['FormattedMessage'].apply(lambda x: len(x.split(" ")))
    df_train, df_test = train_test_split(df, test_size=test_size)
    return df,df_train,df_test