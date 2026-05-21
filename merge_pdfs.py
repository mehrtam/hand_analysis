from pypdf import PdfWriter

pdf1 = "/Users/fateme/Downloads/Share_your_status_with_someone_gov_uk.pdf"
pdf2 = "/Users/fateme/Downloads/StudentVisa.pdf"
output = "Visa_ShareCode.pdf"

merger = PdfWriter()

for pdf in [pdf1, pdf2]:
    merger.append(pdf)

merger.write(output)
merger.close()

print(f"Merged PDF saved as: {output}")
