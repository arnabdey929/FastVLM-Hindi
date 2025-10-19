# FastVLM-Hindi
## This project extends ![Apple's FastVLM](https://github.com/apple/ml-fastvlm) vision-language model by converting the naive English output to Hindi using a small quantized English-to-Hindi translator language model.

The FastVLM model uses a vision encoder and a transformer based language model to caption images live taken on iPhone/Macbook camera. It does a forward pass using the on-device neural engine architecture. The language model used provides captions in English. 

In this project I used an English-to-Hindi translation small language model ![oput-mt-en-hi](https://huggingface.co/Helsinki-NLP/opus-mt-en-hi), which is small enough to do a forward pass on an iPhone while giving excellent translation results.

Since the opus-mt-en-hi model uses its own tokenizer and transformer architecture, the tokenizer, encoder, and the decoder had to be separately quantized and converted to coreML models.

Tokenizer:
- The opus-mt-en-hi uses Marian tokenization, which is simply the [SentencePiece tokenizer](https://github.com/google/sentencepiece),  and uses its own Vocab for token to embedding ID mappings.
- The SentencePiece tokenizer is available as a Swift package in XCode.
- The mapping from tokens to IDs and vice-versa is done using the vocab.json file provided in opus-mt-en-hi repo.

Encoder and Decoder:
- The whole opus-mt-en-hi model showed some errors during coreML conversion, hence separated the encoder and decoder models.
- Created two wrappers for the encoder and decoder, and did a torch trace using dummy inputs.
- During pytorch to coreML conversion, I used flexible input size (1 - 128) for the encoder, but flexible input sizes for the decoder crashed the kernel, hence the encoder has flexible input size, but the decoder has fixed input sizes.
- Since the decoder has fixed input sizes for both the hidden embeddings and the decoder inputs, appropriate padding has to be used.
- As we are only translating the English text to Hindi, accuracy is more important than creativity, hence used greedy decoding by picking the token with highest probability. Also this slightly makes the decoding process faster as compute resources are low here.
- Since we separated out the encoder and decoder, code for autoregressive decoding needed to be manually written in Swift.

App implementation:
- Finally took the original FastVLM model output text and fed through the crafted model.
- Not directly using the output token embeddings of the FastVLM model because the two models use different embeddings for the tokens, also the tokenization scheme might not exactly be the same. Hence manually tokenized the FastVLM output text, mapped to embeddings and then did forward pass.
