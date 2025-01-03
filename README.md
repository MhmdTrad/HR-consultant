### README: AI-Powered HR Consultant Chatbot

---

# AI-Powered HR Consultant Chatbot

This interactive chatbot provides users with expert-level consultation on human resources (HR) topics. Built using LangChain, FAISS, and Google Generative AI (`gemini-pro`), the app is designed to simulate a professional HR consultant that answers user queries accurately and effectively in Arabic or English. The chatbot can retrieve relevant data from uploaded CSV files, process queries, and provide insightful responses, ensuring a seamless and valuable interaction.

---

## Features

1. **AI-Driven HR Expertise**:
   - Offers professional and contextual HR-related advice based on user questions.
   - Differentiates between relevant and non-relevant queries, guiding users appropriately.

2. **Conversational Memory**:
   - Maintains a history of interactions to create a conversational, context-aware experience.

3. **Multilingual Support**:
   - Handles Arabic and English queries, ensuring responses match the user's preferred language.

4. **Knowledge Base Integration**:
   - Employs a FAISS vector database for fast and precise retrieval of HR-related documents.

5. **Interactive Interface**:
   - User-friendly chatbot interface using Streamlit and `streamlit-chat` for an engaging experience.

6. **Customizable Prompt Engineering**:
   - Implements tailored system prompts to ensure accurate, professional responses in HR scenarios.

---

## Key Benefits

1. **Enhanced Productivity**:
   - Automates the process of HR consulting, saving time for both users and HR professionals.

2. **Accurate and Reliable**:
   - Ensures responses are contextually relevant, leveraging AI and preloaded HR knowledge bases.

3. **Dynamic Context Awareness**:
   - Retains conversational history, offering a natural and efficient dialogue.

4. **Scalable and Extendable**:
   - Allows easy expansion by adding new HR documents or enhancing the knowledge base.

5. **Multilingual Capability**:
   - Serves a diverse user base with its Arabic and English language support.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/hr-consultant-chatbot.git
   cd hr-consultant-chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Create a `.env` file and add your Google API key:
     ```
     GOOGLE_API_KEY=your_google_api_key
     ```

4. Load the FAISS vector store:
   - Ensure the `vectorstore.faiss` file exists in the project directory. If not, create it using relevant HR-related CSV files.

5. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## Usage

1. Open the application in your browser.
2. Interact with the chatbot by entering your HR-related queries.
3. View responses in real time and continue the conversation seamlessly.

---

## Example Usage

### User Query (in Arabic):
```
ما هي أفضل الممارسات في عملية تقييم الأداء؟
```

### Response:
```
أفضل الممارسات في عملية تقييم الأداء تشمل:
- تحديد أهداف الأداء بشكل واضح وشفاف.
- استخدام معايير قابلة للقياس لتقييم الموظفين.
- إجراء تقييم منتظم، سواء كان شهريًا أو ربع سنوي.
- تقديم تغذية راجعة بناءة للموظفين بعد التقييم.
- دمج التدريب والتطوير كجزء من عملية التقييم.
```

---

## Requirements

- Python 3.8+
- Streamlit
- Google Generative AI (`gemini-pro`) access
- FAISS
- HuggingFace Embeddings
- Internet connection

---

## Contributing

Contributions are welcome! Submit pull requests or issues for enhancements, new features, or bug fixes.
