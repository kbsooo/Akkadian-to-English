# Deep Past Challenge - Translate Akkadian to English

## Overview
The Deep Past Challenge asks a bold question: Can AI decode 4,000-year-old business records?

In this competition, you will help decode the everyday business records of ancient Assyrian merchants. Using data from 8,000 cuneiform texts, your goal is to build a translation system for Old Assyrian. Thousands more like them lie unread in museum drawers around the world.

Your work can help bring their voices back into the story of humanity.

## Description
Four thousand years ago, Assyrian merchants left behind one of the world’s richest archives of everyday and commercial life. Tens of thousands of clay tablets record debts settled, caravans dispatched, and discuss day-to-day family matters. Today, half of these tablets remain silent, not because they’re damaged, but because so few people can read the language pressed into their clay. Many have sat untranslated in museum drawers for more than a century.

The Deep Past Challenge turns this ancient mystery into a modern machine-learning problem by inviting competitors to help unlock the largest untranslated archive of the ancient world. We invite you to build translation models for Old Assyrian cuneiform tablets: Bronze Age texts that have sat unread in museum collections for over a century. Old Assyrian—the dialect used on these tablets—is an early form of Akkadian, the oldest documented Semitic language.

Nearly twenty-three thousand tablets survive documenting the Old Assyrian trading networks that connected Mesopotamia to Anatolia. Only half have been translated, and less than a dozen scholars in the world are specialized to read the rest.

These aren’t the polished classics of Greece and Rome, curated and copied by scribes who chose whose voices survived. These are unfiltered, straight from the people who wrote them: letters, invoices and contracts written on clay by ancient merchants and their families. They’re the Instagram stories of the Bronze Age: mundane, immediate, and breathtakingly real.

Your task is to build neural machine-translation models that convert transliterated Akkadian into English. The challenge: Akkadian is a low-resource, morphologically complex language where a single word can encode what takes multiple words in English. Standard architectures built for modern, data-rich languages fail here. Crack this problem and you’ll give voice to 10,000+ untranslated tablets. And you'll do more than revive the past: you'll help pioneer a blueprint for translating the thousands of endangered and overlooked languages—ancient and modern—that the AI age has yet to reach.

Visit this website to learn more about the organizing team, the Deep Past Initiative, and to find more background materials.

## Evaluation
Submissions are evaluated by the Geometric Mean of the BLEU and the chrF++ scores, with each score's sufficient statistics being aggregated across the entire corpus (that is, each score is a micro-average).

You may refer to the SacreBLEU library for implementation details. A notebook implementing the metric on Kaggle may be found here: Geometric Mean of BLEU and chrF++.

## Submission File
For each id in the test set, you must predict an English translation of the associated Akkadian transliteration. Each translation should comprise a single sentence. The file should contain a header and have the following format:

```
id,translation
0,Thus Kanesh, say to the -payers, our messenger, every single colony, and the...
1,In the letter of the City (it is written): From this day on, whoever buys meteoric...
2,As soon as you have heard our letter, who(ever) over there has either sold it to...
3,Send a copy of (this) letter of ours to every single colony and to all the trading...
...
```