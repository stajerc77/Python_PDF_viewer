# build text summarizer for the PDF viewer that works with a local LLM file in GGUF format.
# apply chunking such that the LLM max tokens are not exceeded


def build_summarizer(llm):
    def summarize(text, chunk_size=1500):
        # split text into chunks small enough for the model
        words = text.split()
        chunks = []
        current = []

        for word in words:
            current.append(word)
            if len(current) >= chunk_size:
                chunks.append(" ".join(current))
                current = []
        if current:
            chunks.append(" ".join(current))

        summaries = []

        for chunk in chunks:
            prompt = f"Summarize the following text in a concise paragraph of 5 to 7 sentences:\n\n{chunk}\n\nSummary:"
            
            out = llm(
                prompt,
                max_tokens=300,
                temperature=0.2,
            )

            summary = out["choices"][0]["text"].strip()
            summaries.append(summary)

        final_prompt = (
            "Combine the following partial summaries into a single coherent summary:\n\n"
            + "\n\n".join(summaries)
            + "\n\nFinal summary:"
        )

        final_output = llm(final_prompt, max_tokens=300, temperature=0.2)
        final_summary = final_output["choices"][0]["text"].strip()

        return final_summary

    return summarize
