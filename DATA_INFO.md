# Dataset Instructions

By far the biggest challenge in working with Akkadian / Old Assyrian texts is dealing with the formatting issues. As they say, “garbage in, garbage out” and unfortunately, the format of text in transliteration poses challenges at each step of the ML workflow, from tokenization to the transformation and embedding process.

To mitigate these issues, we provide the following information and suggestions in handling the different formatting challenges in both the transliterated and translated texts.

## Texts in Transliteration

- Main formatting challenges: in addition to the standard transliteration format, with hyphenated syllables, additional scribal additions have encumbered the text with superscripts, subscripts, and punctuations only meaningful to specialists in Assyriology (Complete Transliteration Conversion Guide).
- Capitalization is also a challenge, as it encodes meaning in two different ways. When the first letter of a word is capitalized it implies the word is a personal name or a place name (i.e. proper noun). When the word is in ALL CAPS, that implies it is a Sumerian logogram and was written in place of the Akkadian syllabic spelling for scribal simplicity.
- Determinatives are used in Akkadian as a type of classifier for nouns and proper nouns. These signs are usually printed in superscript format adjacent to the nouns they classify. To avoid the potential confusion of reading a determinative sign as part of a work, we have followed the standard transliteration guide and retained curly brackets around these. While this may pose challenges in ML, we note that this is the only use of curly brackets in the transliteration (e.g. a-lim{ki}, A-mur-{d}UTU).
- Broken text on the tablet: as these are ancient texts, they include a number of breaks and lacunae. In order to standardize these breaks, we suggest using only two markers, one for a small break of a single sign <gap> and the other for more than one sign up to large breaks <big_gap>.
- For the purpose of this challenge, we include suggestions of how best to handle these formatting issues below.

## Texts in Translation

There is currently no complete or extensive database for translations of ancient cuneiform documents, and this is especially true for the Old Assyrian texts. For this reason, we gathered together the books and articles with the translations and commentaries of the Old Assyrian texts and we digitized them with an OCR and LLM for corrections. Even after all that work, there are still a number of formatting issues with these translations, which makes this a central component of the challenge for successful machine translation development.

Translations usually retain the same proper noun capitalization, and these proper nouns in general are where most ML tasks underperform. To account for these issues, we have included a lexicon in the dataset which includes all the proper nouns as specialists have normalized them for print publications.

### Modern Scribal Notations
Lastly, it is important to note that there are modern scribal notations that accompany the text in transliteration and translation. The first of these include line numbers. These are typically numbered 1, 5, 10, 15, etc. However, if there are any broken lines, the line numbers will have an apostrophe immediately following (‘), and if there is a second set of broken lines, the line numbers will have two trailing apostrophes (‘’). These are not quotation marks, but a scribal convention editors sometimes use in publication.

Additional scribal notations include:

- Exclamation marks when a scholar is certain about a difficult reading of a sign !
- Question mark when a scholar is uncertain about a difficult reading of a sign ?
- Forward slash for when the signs belonging to a line are found below the line /
- Colon for the Old Assyrian word divider sign :
- Comments for breaks and erasures in parentheses ( )
- Scribal insertions when a correction is made in pointy brackets < >
- The demarcation of errant or erroneous signs in double pointy brackets << >>
- Half brackets for partially broken signs ˹ ˺
- Square brackets for clearly broken signs and lines [ ]
- Curly brackets for determinatives (see below) { }

## Formatting Suggestions for Transliterations and Translations:

### Remove (modern scribal notations):
- ! (certain reading)
- ? (questionable reading)
- / (line divider)
- : OR . (word divider)
- < > (scribal insertions, but keep the text in translit / translations)
- ˹ ˺ (partially broken signs, to be removed from transliteration)
- [ ] (remove from document level transliteration. e.g. [KÙ.BABBAR] → KÙ.BABBAR)

### Replace (breaks, gaps, superscripts, subscripts):
- [x] <gap>
- … <big_gap>
- [… …] <big_gap>
- ki {ki} (see full list below)
- il5 il5 (same for any subscripted number)

### Additional Characters & Formats (you may encounter):

| Character | CDLI | ORACC | Unicode |
|---|---|---|---|
| á | a2 | a₂ | U+2082 (subscript 2) |
| à | a3 | a₃ | U+2083 (subscript 3) |
| é | e2 | e₂ | U+2082 |
| è | e3 | e₃ | U+2083 |
| í | i2 | i₂ | U+2082 |
| ì | i3 | i₃ | U+2083 |
| ú | u2 | u₂ | U+2082 |
| ù | u3 | u₃ | U+2083 |
| š | sz | š | U+0161 |
| Š | SZ | Š | U+0160 |
| ṣ | "s," | ṣ | U+1E63 |
| Ṣ | "S," | Ṣ | U+1E62 |
| ṭ | "t," | ṭ | U+1E6D |
| Ṭ | "T," | Ṭ | U+1E6C |
| ‘ | ‘ | ʾ | U+02BE |
| 0-9 | 0-9 | ₀-₉ (subscript) | U+2080-U+2089 |
| xₓ | Xx | ₓ (subscript x) | U+2093 |
| ḫ | h | ḫ | U+1E2B |
| Ḫ | H | Ḫ | U+1E2A |

These rows of Ḫ ḫ are here to indicate that training data (and publication data) has Ḫ ḫ but the test data has only H h.

There is only one type of H in Akkadian, so this can be a simple substitution for transliteration text Ḫ ḫ --> H h

### Akkadian determinatives in curly brackets:
1. {d} = dingir ‘god, deity’ — d preceding non-human divine actors
2. {mul} = ‘stars’ — MUL preceding astronomical bodies and constellations
3. {ki} = ‘earth’ — KI following a geographical place name or location
4. {lu₂} = LÚ preceding people and professions
5. {e₂} = {É} preceding buildings and institutions, such as temples and palaces
6. {uru} = (URU) preceding names of settlements, such as villages, towns and cities
7. {kur} = (KUR) preceding lands and territories as well as mountains
8. {mi} = munus (f) preceding feminine personal names
9. {m} = (1 or m) preceding masculine personal names
10. {geš} / {ĝeš) = (GIŠ) preceding trees and things made of wood
11. {tug₂} = (TÚG) preceding textiles and other woven objects
12. {dub} = (DUB) preceding clay tablets, and by extension, documents and legal records
13. {id₂} = (ÍD) (a ligature of A and ENGUR, transliterated: A.ENGUR) preceding names of canals or rivers or when written on its own referring to the divine river
14. {mušen} = (MUŠEN) preceding birds
15. {na₄} = (na4) preceding stone
16. {kuš} = (kuš) preceding (animal) skin, fleece, hides
17. {u₂} = (Ú) preceding plants

## Citation
Abdulla, F., Agarwal, R., Anderson, A., Barjamovic, G., Lassen, A., Ryan Holbrook, and María Cruz. Deep Past Challenge - Translate Akkadian to English. https://kaggle.com/competitions/deep-past-initiative-machine-translation, 2025. Kaggle.

# Dataset Description

The competition data comprises transliterations of over 8,000 Old Assyrian cuneiform texts with a comprehensive set of metadata. We provide aligned English translations for a subset of these. We also provide unprocessed texts from almost 900 scholarly publications containing more translations from which you may attempt to create additional training data.

See also the Dataset Instructions for more on the formatting conventions used in these transliterations.

Please note that this is a Code Competition. The data in test.csv is only dummy data to help you author your solutions. When your submission is scored, this example test data will be replaced with the full test set.

## File and Field Information
- `train.csv` - About 1500 transliterations of Old Assyrian texts from the original excavated tablets each with a translation into English.

    - `oare_id` - Identifier in the Old Assyrian Research Environment (OARE) database. Uniquely identifies each text.
    - `transliteration` - An Akkadian transliteration of the original tablet text.
    - `translation` - A corresponding English translation.
- `test.csv` - A small example set representative of the test data. When your submission is scored, this example test data will be replaced with the full test set. There are about 4000 sentences in the test data from about 400 unique documents. Note that while the training data has translations aligned at the document level, the test data has translations aligned at the sentence level.

    - `id` - A unique identifier for each sentence.
    - `text_id` - A unique identifier for each document.
    - `line_start, line_end` - Denotes the boundaries of the sentence within the original tablet. Orders the sentences within the document. Note that this field has `str` type with values like `1`, `1'`, or `1''`. See the note on line numbers under Modern Scribal Notations in the Dataset Instructions.
    - `transliteration` - An Akkadian transliteration of the original tablet text. Your goal is to produce the corresponding `translation`.
- `sample_submission.csv` - A sample submission file in the correct format. See the Evaluation page for more details.

### Supplemental Data

- `published_texts.csv` - About 8,000 transliterations of Old Assyrian texts together with metadata fields and catalog information from database and museum records as published in the OARE database. You may use these identifiers to retrieve additional information from the linked websites. These transliterations are not provided with translations.

    - `oare_id` - Identifier in the OARE database, as in `train.csv`.
    - `online transcript` - URL of the transliteration transcript hosted on the DPI website.
    - `cdli_id` - Identifier in the CDLI website. Multiple IDs are separated by bar `|`.
    - `aliases` - Other published labels for the text (e.g. publication numbers, museum IDs, etc.). Multiple IDs are separated by bar `|`.
    - `label` - Primary designation as a label of the text.
    - `publication_catalog` - Labels of the text found in publications and museum records. Multiple IDs are separated by bar `|`.
    - `description` - Basic description of the text.
    - `genre_label` - Basic genre assigned to the text. Not available for all texts.
    - `inventory_position` - Label of the text as found in the museum. Multiple IDs are separated by bar `|`.
    - `online_catalog` - URL of the Yale collection with CC-0 metadata and images.
    - `note` - Notes made by specialists for commentary or translations.
    - `interlinear_commentary` - References to publications which discuss the text at specific lines.
    - `online_information` - URL of the text in the British Museum (note these images are copyright of the British Museum, not in CC). Not available for all texts.
    - `excavation_no` - Identifier assigned to the text from the excavation.
    - `oatp_key` - Identifier assigned by the Old Assyrian Text Project.
    - `eBL_id` - Identifier in the eBL website.
    - `AICC_translation` - URL of the first published online machine translation. Note that most of these translations are very poor quality.
    - `transliteration_orig` - Original text transliteration from the OARE database.
    - `transliteration` - Clean version of the text transliteration based on these formatting suggestions.
- `publications.csv` - Contains the raw text of about 880 scholarly publications containing translations from Old Assyrian into multiple modern languages. The texts were produced via OCR with LLM postprocessing. You may attempt to extract these translations and align them with the transliterations in `published_texts.csv`. Note that the translations often are given in a language other than English.
    - `pdf_name` - The name of the PDF file from which the text was extracted.
    - `page` - The page number where the given text occured.
    - `page_text` - The text of the article itself.
    - `has_akkadian` - Whether or not the text contains Akkadian transliterations.
- `bibliography.csv` - Bibliographic data for the texts in `publications.csv`.
    - `pdf_name` - An ID corresponding to that in `publications.csv`.
    - `title`, `author`, `author_place`, `journal`, `volume`, `year`, `pages` - Standard bibliographic data.
- `OA_Lexicon_eBL.csv` - This file contains a list of all the Old Assyrian words in transliteration with their lexical equivalents (that is, how they are found in a dictionary). The links included are to an online Akkadian dictionary hosted by LMU, the electronic Babylonian Library (eBL).
    - `type` - Type of word (e.g. word, `PN` = person name, `GN` = geographic name).
    - `form` - String-literal word, as found in transliteration.
    - `norm` - Normalized form, with hyphens removed and vowel length indications.
    - `lexeme` - Lemmatized form, as found in a dictionary.
    - `eBL` - URL of the online dictionaries in the electronic Babylonian Library (eBL).
    - `I_IV` - Roman numeral designation of the homonym lexemes, corresponding to the Concise Dictionary of Akkadian (CDA) found in the eBL.
    - `A_D` - Alphabetic designation of the homonym lexemes, corresponding to the Chicago Assyrian Dictionary.
    - `Female(f)` - Designation for female gender.
    - `Alt_lex` - Alternative normalizations.
- `eBL_Dictionary.csv` - The complete dictionary of Akkadian words from the eBL database. It collects the data provided by the URLs at `eBL` in the `OA_Lexicon_eBL.csv` file.
- `resources.csv` - A list of resources that might be used for additional data.
- `Sentences_Oare_FirstWord_LinNum.csv` - An aid to aligning translations at the sentence level for the data in `train.csv`. Indicates the first word of each sentence and its location on the tablet.

### Suggested Workflow for Building Additional Training Data
The `publications.csv` file contains OCR output from almost 900 PDFs, and extracting the translations from these texts is an essential first step. Before any machine learning can happen, the training data needs to be reconstructed and aligned. Here’s a simple path you can follow:

1. Locate each text and its translation: Use the document identifiers (IDs, aliases, or museum numbers) to match transliterations with their corresponding translations in the OCR output.

2. Convert all translations to English: The source translations may appear in multiple languages (e.g., English, French, German, Turkish). For consistency, convert everything to English.

3. Create sentence-level alignments: Break both the Akkadian transliteration and the matching English translation into sentences and align them pairwise. This sentence-level mapping is the most useful format for training and evaluating machine translation models.

Once these steps are completed, you’ll have a clean, aligned dataset ready for machine learning.

## Bibliography
The bibliography reflects the secondary sources we used to retrieve the translations for the challenge. Because they are held in different copyrights, we suggest each work should be cited if they were used when generating machine translations.

Additional bibliography citations for the primary sources can be found here:

https://cdli.earth/publications
https://cdli.ox.ac.uk/wiki/abbreviations_for_assyriology