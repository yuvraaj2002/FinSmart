from urllib.request import urlopen
import certifi
import json
import pandas as pd
from src.logger import logging
from rich import print

from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from src.logger import logging



class FinancialDataProcessor:

    def __init__(self, api_key,ticker,exchange):
        self.api_key = api_key
        self.ticker = "MSFT"
        self.exchange = "US"
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=5)

        # Setup for the embedding model
        self.model_name = "BAAI/bge-large-en-v1.5"
        self.model_kwargs = {'device': 'cpu'}
        self.encode_kwargs = {'normalize_embeddings': True}
        self.hf = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )

        # Setup for the vector database
        self.qdrant_url = "https://localhost:6333"
        self.collection_name = "finsmart_vdb"
        logging.info("Huggingface model and vector database parameters setup succesfully")


    def get_json_parsed_data(self):
        """
        Fetches and parses JSON data from Financial Modeling Prep API based on the exchange and ticker symbol.

        Parameters:
        - exchange (str): The stock exchange to query (e.g., "NSE" for National Stock Exchange of India).
        - ticker (str): The ticker symbol of the stock to query.

        Returns:
        - dict: The parsed JSON data from the API response.
        """
        if self.exchange == "NSE":
            url = f"https://financialmodelingprep.com/api/v3/search?query={self.ticker}&exchange=NSE&apikey={self.api_key}"
        else:
            url = f"https://financialmodelingprep.com/api/v3/quote/{self.ticker}?apikey={self.api_key}"
        response = urlopen(url, cafile=certifi.where())
        data = response.read().decode("utf-8")
        return json.loads(data)

    def preprocess_csv_data(self, df, csv_path):
        """
        Preprocesses the economic data DataFrame by converting 'timestamp' and 'earningsAnnouncement' columns to datetime format.

        Parameters:
        - df (pd.DataFrame): The DataFrame containing economic data.
        - csv_path (str): The path where the processed CSV file will be saved.

        Returns:
        - pd.DataFrame: The preprocessed DataFrame with 'timestamp' and 'earningsAnnouncement' columns converted to datetime format.
        """
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['earningsAnnouncement'] = pd.to_datetime(df['earningsAnnouncement'])
        df.to_csv(f"{csv_path}/process_data.csv")
        return df

    def laod_split(self,csv_path):

        # Loading the CSV file using CSV loader  and creating the documents
        loader = CSVLoader(csv_path)
        documents = loader.load()
        logging.info("Loaded the csv file using the csv loader")

        # Creating the chunks
        text_chunks = self.text_splitter.split_documents(documents)
        logging.info("Created the chunks")
        return text_chunks



if __name__ == "__main__":

    # Creating an instance of the class
    fin_obj = FinancialDataProcessor(api_key="C1HRSweTniWdBuLmTTse9w8KpkoiouM5",ticker="MSFT",exchange="US")
    
    raw_df = pd.DataFrame(fin_obj.get_json_parsed_data())
    logging.info("Loaded the dataframe using the API")

    fin_obj.preprocess_csv_data(raw_df,r"D:\Projects\FinSmart\Data")
    logging.info("Processed the raw dataframe and stored the csv file")

    text_chunks = fin_obj.laod_split(r"D:\Projects\FinSmart\Data\process_data.csv")
