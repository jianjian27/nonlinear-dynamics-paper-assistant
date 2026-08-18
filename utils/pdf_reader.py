import fitz



def extract_pages_from_pdf(pdf_path):

    """
    提取PDF每一页文本

    返回:

    [
        {
            "page":1,
            "text":"xxxx"
        }
    ]

    """


    pages = []


    doc = fitz.open(pdf_path)


    for index, page in enumerate(doc):


        text = page.get_text()


        pages.append(
            {
                "page": index + 1,
                "text": text
            }
        )


    doc.close()


    return pages