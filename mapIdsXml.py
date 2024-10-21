import xml.etree.ElementTree as ET
import logging
import re
import requests
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


xml_file_path='topics.xml'
csvFilePath='summary.csv'
csvOutputFile='comparision-80.csv'
logging.basicConfig(level=logging.INFO)
apiToken = "f7ae1c7201b9a61fcadedbb2e92c0b77"

tree=ET.parse(xml_file_path)
root=tree.getroot()

ns={
    'wp': 'http://wordpress.org/export/1.2/', 
    'content': 'http://purl.org/rss/1.0/modules/content/',
    'dc': 'http://purl.org/dc/elements/1.1/',
    'excerpt': 'http://wordpress.org/export/1.2/excerpt/'
}

items=root.findall('.//item')
videoIdData={}
for item in items:
    topicId=item.find('wp:post_id', ns)
    # if topicId is not None:
    #     logging.info(topicId.text)
    content=item.find("content:encoded",ns)
    if content is not None:
        # logging.info(content.text)
        if(str(type(content.text))!="<class 'NoneType'>"):
            if("https://vimeo.com" in content.text):
                urlPattern=r'https?://vimeo[^\s"<>]+'
                matches=re.findall(urlPattern, content.text)
                videoIdData[topicId.text]=set(matches)
logging.info(videoIdData)

videoInfo={}
failed=[]
logging.info(videoIdData)
for eachTopic in videoIdData.keys():
    # logging.info(eachTopic)
    # logging.info(videoIdData[eachTopic])
    # if(eachTopic in ['10480201','10480202','10480203','10480204']):
    for eachVideo in videoIdData[eachTopic]:
        eachVideoID=eachVideo.split("/")[-1]
        url = "https://api.vimeo.com/videos/"+eachVideoID
        headers = {
            'Authorization': f'Bearer {apiToken}',
        }
        response = requests.get(url,headers=headers)
        if response.status_code==200:
            # logging.info("Success!")
            # logging.info(response.json()["name"])
            videoInfo[eachTopic]=response.json()["name"]
        else:
            # logging.info(f"Failed with status code {response.status_code}")
            # logging.info(response.text)
            failed.append(eachTopic)
        logging.info(videoInfo)
logging.info(failed)

csvFileInfo={}
with open(csvFilePath,mode='r',encoding='utf-8') as csvFile:
    csvReader=csv.DictReader(csvFile)
    for row in csvReader:
        csvFileInfo[row["ID"]]=row["video_name"]
        # csvFileInfo.append(row["video_name"])

logging.info(csvFileInfo)

with open(csvOutputFile, mode='w', newline='', encoding='utf-8') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow(['Topic_id', 'Erik_id', 'Erik_video_title', 'Video_title','match_score'])

    for learndashId in videoInfo.keys():
        videoTitle=videoInfo[learndashId]
        for erikIds in csvFileInfo.keys():
            erikVideoTitle=csvFileInfo[erikIds]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([videoTitle, erikVideoTitle])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            if(similarity[0][0]>0.8):
                logging.info(learndashId+" "+videoTitle+" "+erikIds+" "+erikVideoTitle+" "+str(similarity[0][0]))
                writer.writerow([learndashId,erikIds,erikVideoTitle,videoTitle,format(similarity[0][0]*100,".2f")])
