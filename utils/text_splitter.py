def split_pages(
        pages,
        chunk_size=4000,
        overlap=500
):


    chunks = []


    current_text = ""

    current_pages = []


    for page in pages:


        current_text += page["text"]

        current_pages.append(
            page["page"]
        )


        if len(current_text) >= chunk_size:


            chunks.append(
                {
                    "text": current_text,

                    "pages": current_pages
                }
            )


            # 保留部分重叠

            current_text = current_text[-overlap:]


            current_pages = [
                current_pages[-1]
            ]



    # 最后一部分

    if current_text:


        chunks.append(
            {
                "text": current_text,

                "pages": current_pages
            }
        )


    return chunks