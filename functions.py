import requests
import json
import polars as pl
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
import os

def getVideoRecords(response: requests.models.Response) -> list:
    """
        Function to extract YouTube video data from GET request response

        Dependers: 
            - getVideoIDs()
    """

    video_record_list = []
    
    for raw_item in json.loads(response.text)['items']:
    
        # only execute for youtube videos
        if raw_item['id']['kind'] != "youtube#video":
            continue
        
        video_record = {}
        video_record['video_id'] = raw_item['id']['videoId']
        video_record['datetime'] = raw_item['snippet']['publishedAt']
        video_record['title'] = raw_item['snippet']['title']
        
        video_record_list.append(video_record)

    return video_record_list


def getVideoIDs():
    """
        Function to return all video IDs for Shaw Talebi's YouTube channel

        Dependencies: 
            - getVideoRecords()
    """


    channel_id = 'UCa9gErQ9AE5jT2DZLjXBIdA' # my YouTube channel ID
    page_token = None # initialize page token
    url = 'https://www.googleapis.com/youtube/v3/search' # YouTube search API endpoint
    my_key = os.getenv('YT_API_KEY')

    # extract video data across multiple search result pages
    video_record_list = []

    while page_token != 0:
        params = {"key": my_key, 'channelId': channel_id, 'part': ["snippet","id"], 'order': "date", 'maxResults':50, 'pageToken': page_token}
        response = requests.get(url, params=params)

        # append video records to list
        video_record_list += getVideoRecords(response)

        try:
            # grab next page token
            page_token = json.loads(response.text)['nextPageToken']
        except:
            # if no next page token kill while loop
            page_token = 0

    # write videos ids as parquet file
    pl.DataFrame(video_record_list).write_parquet('data/video-ids.parquet')


def extractTranscriptText(transcript: list) -> str:
    """
        Function to extract text from transcript dictionary

        Dependers:
            - getVideoTranscripts()
    """
    
    text_list = [transcript[i]['text'] for i in range(len(transcript))]
    return ' '.join(text_list)


def getVideoTranscripts():
    """
        Function to extract transcripts for all video IDs stored in "data/video-ids.parquet"

        Dependencies:
            - extractTranscriptText()
    """


    df = pl.read_parquet('data/video-ids.parquet')

    transcript_text_list = []

    for i in range(len(df)):

        # try to extract captions
        try:
            transcript = YouTubeTranscriptApi.get_transcript(df['video_id'][i])
            transcript_text = extractTranscriptText(transcript)
        # if not available set as n/a
        except:
            transcript_text = "n/a"
        
        transcript_text_list.append(transcript_text)

    # add transcripts to dataframe
    df = df.with_columns(pl.Series(name="transcript", values=transcript_text_list))

    # write dataframe to file
    df.write_parquet('data/video-transcripts.parquet')


def handleSpecialStrings(df: pl.dataframe.frame.DataFrame) -> pl.dataframe.frame.DataFrame:
    """
        Function to replace special character strings in video transcripts and titles
        
        Dependers:
            - transformData()
    """

    special_strings = ['&#39;', '&amp;', 'sha ']
    special_string_replacements = ["'", "&", "Shaw "]

    for i in range(len(special_strings)):
        df = df.with_columns(df['title'].str.replace(special_strings[i], special_string_replacements[i]).alias('title'))
        df = df.with_columns(df['transcript'].str.replace(special_strings[i], special_string_replacements[i]).alias('transcript'))

    return df

def setDatatypes(df: pl.dataframe.frame.DataFrame) -> pl.dataframe.frame.DataFrame:
    """
        Function to change data types of columns in polars data frame containing video IDs, dates, titles, and transcripts

        Dependers:
            - transformData()
    """

    # change datetime to Datetime dtype
    df = df.with_columns(pl.col('datetime').cast(pl.Datetime))

    return df


def transformData():
    """
        Function to preprocess video data

        Dependencies:
            - handleSpecialStrings()
            - setDatatypes()
    """

    df = pl.read_parquet('data/video-transcripts.parquet')

    df = handleSpecialStrings(df)
    df = setDatatypes(df)

    df.write_parquet('data/video-transcripts.parquet')

def createTextEmbeddings():
    """
        Function to generate text embeddings of video titles and transcripts
    """

    # read data from file
    df = pl.read_parquet('data/video-transcripts.parquet')

    # define embedding model and columns to embed
    # model_path = 'data/all-MiniLM-L6-v2'
    # model = SentenceTransformer(model_path)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    column_name_list = ['title', 'transcript']

    for column_name in column_name_list:
        # generate embeddings
        embedding_arr = model.encode(df[column_name].to_list())

        # store embeddings in a dataframe
        schema_dict = {column_name+'_embedding-'+str(i): float for i in range(embedding_arr.shape[1])}
        df_embedding = pl.DataFrame(embedding_arr, schema=schema_dict)

        # append embeddings to video index
        df = pl.concat([df, df_embedding], how='horizontal')

    # write data to file
    df.write_parquet('data/video-index.parquet')