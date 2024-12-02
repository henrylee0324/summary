import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime


load_dotenv()

class Summary:
    def __init__(self, article_path="articles.csv", output_path="articles_with_summary.csv"):
        self.api_key = os.getenv("API_KEY")
        self.model_name = os.getenv("LLM_MODEL")
        self.article_path = article_path
        self.output_path = output_path
        self.retry_attempts = 3
        self.retry_delay = 5  # seconds

        # 檢查和載入文章資料
        self.articles = pd.read_csv(self.article_path)
        if 'ARTICLE_TEXT' not in self.articles.columns:
            raise ValueError("The CSV file does not contain an 'ARTICLE_TEXT' column.")
        
    def add_summaries(self):
        summaries = []
        for index, row in self.articles.iterrows():
            article_text = row['ARTICLE_TEXT']
            print(f"Generating summary for article {index + 1}/{len(self.articles)}")
            summary = self._generate_summary(article_text)
            summaries.append(summary)
        
        # 加入摘要到 DataFrame 並輸出為新的 CSV
        self.articles['summary'] = summaries
        self.articles.to_csv(self.output_path, index=False)
        print(f"Summaries saved to {self.output_path}")

    def _generate_summary(self, article):
        # 建立摘要提示
        summary_prompt = f"""
        You are tasked with summarizing a news article in a concise and informative manner. Here's the article you need to summarize:

        <article>
        {article}
        </article>

        Please read the article carefully and create a summary following these guidelines:

        1. Identify the main points and key information from the article.
        2. Summarize the article using a maximum of 5 bullet points.
        3. Each bullet point should be a single sentence that captures an important aspect of the article.
        4. Ensure that the bullet points cover the most crucial information and provide a comprehensive overview of the article's content.
        5. Focus on facts and avoid including personal opinions or interpretations.

        Provide your summary within <summary> tags, with each bullet point enclosed in <bullet> tags.
        """
        return self._call_llm("", summary_prompt)

    def _call_llm(self, system_prompt, user_prompt):
        attempts = 0
        while attempts < self.retry_attempts:
            try:
                # 初始化 OpenAI 客戶端
                client = OpenAI(api_key=self.api_key)
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                return completion.choices[0].message.content
            except Exception as e:
                attempts += 1
                print(f"Error calling LLM API (Attempt {attempts}/{self.retry_attempts}): {e}")
                if attempts < self.retry_attempts:
                    time.sleep(self.retry_delay)  # 延遲後重試
                else:
                    raise RuntimeError(f"Failed to call LLM API after {self.retry_attempts} attempts.") from e


if __name__ == "__main__":
    # 執行摘要生成流程
    summary = Summary()
    summary.add_summaries()
