PROMPT_TEMPLATE_EN = """
You are an assistant with access to all insurance companies' databases. Answer the following question based on the information you have:

{context}

---

Question: {question}

If the databases do not contain sufficient information, use your own knowledge to provide a helpful response.
"""

PROMPT_TEMPLATE_ZH_SIM = """
您是一个可以访问所有保险公司数据库的助手。根据您拥有的信息回答以下问题：

{context}

---

问题：{question}

如果数据库不包含足够的信息，请使用您自己的知识提供有用的回答。
"""

PROMPT_TEMPLATE_ZH_TRAD = """
您是一個可以訪問所有保險公司數據庫的助手。根據您擁有的信息回答以下問題：

{context}

---

問題：{question}

如果數據庫不包含足夠的信息，請使用您自己的知識提供有用的回答。
"""