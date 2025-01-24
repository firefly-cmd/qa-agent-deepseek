from retrieve import DocumentRetriever
import ollama
import regex as re

class QAPipeline:
    def __init__(self):
        self.retriever = DocumentRetriever()
        
    PROMPT_TEMPLATE = """Context information:
        {context}

        Using the context above and your general knowledge, answer this question:
        Question: {question}

        Format requirements:

        - If uncertain, say "The documents don't specify"""

    def parse_response(self, response: str) -> dict:
        """Extract thinking and answer components without <answer> tags"""
        # Extract thinking process
        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        
        # Get everything AFTER </think> as the answer
        answer_start = response.find('</think>') + len('</think>')
        answer = response[answer_start:].strip()
        
        return {
            "thinking": think_match.group(1).strip() if think_match else "",
            "answer": answer,
            "raw_response": response
        }

    def generate_answer(self, question: str, k: int = 5) -> dict:
        """Full QA workflow with enhanced output"""
        try:
            # Retrieve documents
            context_docs = self.retriever.query_documents(question, k=k)
            
            # Format context preserving full metadata
            context_str = "\n".join(
                f"[Document {idx+1}] {doc['source']} (Page {doc['page']}):\n{doc['text']}"
                for idx, doc in enumerate(context_docs)
            )
            
            # Generate response
            response = ollama.generate(
                model="deepseek-r1:latest",
                prompt=self.PROMPT_TEMPLATE.format(
                    context=context_str,
                    question=question
                )
            )
            
            # Parse components
            parsed = self.parse_response(response['response'])
            
            return {
                **parsed,
                "sources": [
                    {
                        "source": doc["source"],
                        "page": doc["page"],
                        "confidence": doc["score"],
                        "full_text": doc["text"]
                    } for doc in context_docs
                ]
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "thinking": "",
                "answer": "Failed to generate response",
                "sources": []
            }

