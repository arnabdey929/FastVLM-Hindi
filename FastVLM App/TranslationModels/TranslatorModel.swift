import Foundation
import CoreML

class TranslatorModel {
    
    var encoderModel: MLModel?
    var decoderModel: MLModel?
    let seqLen = 128
    let vocabSize = 61950
    let startTokenId = 61949 // <pad>: 61949, Marian decoder's start token for decoder
    let eosTokenID = 0 // </s>
    let tokeniser = Tokeniser()
    
    init() {
        
        guard let encoderURL = Bundle.main.url(forResource: "MarianEncoder", withExtension: "mlmodelc") else {
            fatalError("Could not find MarianEncoder.mlpackage")
        }
        
        guard let decoderURL = Bundle.main.url(forResource: "MarianDecoder", withExtension: "mlmodelc") else {
            fatalError("MarianDecoder model not found")
        }
        
        do {
            encoderModel = try MLModel(contentsOf: encoderURL)
            decoderModel = try MLModel(contentsOf: decoderURL)
        } catch {
            print("Failed to load the models: \(error.localizedDescription)")
        }
    }
    
    func makeMultiArrayInt(ints: [Int]) -> MLMultiArray {
        let shape: [NSNumber] = [1, NSNumber(value: ints.count)]
        guard let array = try? MLMultiArray(shape: shape, dataType: .int32) else {
            fatalError("Failed to create MLMultiArray")
        }
        for (i, val) in ints.enumerated() {
            array[[0, NSNumber(value: i)]] = NSNumber(value: Int32(val))
        }
        return array
    }
    
    func makeMultiArray(ints: [Int]) -> MLMultiArray {
        let shape: [NSNumber] = [1, NSNumber(value: ints.count)]
        guard let array = try? MLMultiArray(shape: shape, dataType: .float32) else {
            fatalError("Failed to create MLMultiArray")
        }
        for (i, val) in ints.enumerated() {
            array[[0, NSNumber(value: i)]] = NSNumber(value: Float(val))
        }
        return array
    }
    
    func padHidden(_ array: MLMultiArray, toLength targetLength: Int) -> MLMultiArray {
        let shape = array.shape.map { $0.intValue }
        guard shape.count == 3 else {
            print("Expected 3D MultiArray")
            return makeMultiArray(ints: [0])
        }
        
        let batch = shape[0]
        let seqLen = shape[1]
        let hiddenDim = shape[2]
        
        guard targetLength >= seqLen else {
            print("Target length smaller than current sequence length")
            return makeMultiArray(ints: [0])
        }
        
        guard let padded = try? MLMultiArray(shape: [NSNumber(value: batch),
                                                    NSNumber(value: targetLength),
                                                     NSNumber(value: hiddenDim)], dataType: array.dataType) else {
            return makeMultiArray(ints: [0])
        }
        
        for i in 0..<padded.count {
            padded[i] = NSNumber(value: 0)
        }
        
        for b in 0..<batch {
            for t in 0..<seqLen {
                for h in 0..<hiddenDim {
                    let srcIdx = b * seqLen * hiddenDim + t * hiddenDim + h
                    let dstIdx = b * targetLength * hiddenDim + t * hiddenDim + h
                    padded[NSInteger(dstIdx)] = array[NSInteger(srcIdx)]
                }
            }
        }
        
        return padded
    }
    
    func padDecoderInput(_ array: MLMultiArray, toLength targetLength: Int) -> MLMultiArray? {
        let shape = array.shape.map { $0.intValue }
        guard shape.count == 2 else {
            print("Expected 2D MultiArray")
            return nil
        }
        
        let batch = shape[0]
        let seqLen = shape[1]
        
        guard targetLength >= seqLen else {
            print("Target length smaller than current sequence length")
            return nil
        }
        
        guard let padded = try? MLMultiArray(shape: [NSNumber(value: batch),
                                                     NSNumber(value: targetLength)], dataType: array.dataType) else {
            return nil
        }
        
        for i in 0..<padded.count {
            padded[i] = NSNumber(value: startTokenId)
        }
        
        for b in  0..<batch {
            for t in 0..<seqLen {
                let srcIdx = b * seqLen + t
                let dstIdx = b * targetLength + t
                padded[dstIdx] = array[srcIdx]
            }
        }
        
        return padded
    }
    
    func convertToFloat32(_ array: MLMultiArray) -> MLMultiArray {
        guard let converted = try? MLMultiArray(shape: array.shape, dataType: .float32) else {
            print("Failed to convert to float32 from float16")
            return makeMultiArray(ints: [0])
        }
        
        for i in 0..<array.count {
            converted[i] = NSNumber(value: Float(array[i].floatValue))
        }
        return converted
    }
    
    func mlMultiArrayToIntArray(_ array: MLMultiArray) -> [Int] {
        let length = array.shape[1].intValue
        var intArray: [Int] = []
        
        for i in 0..<length {
            let val = array[[0, NSNumber(value: i)]].intValue
            intArray.append(val)
        }
        return intArray
    }
    
    func sliceMLMultiArray(_ array: MLMultiArray, upTo index: Int) -> MLMultiArray? {
        let seqLen = array.shape[1].intValue
        let newLen = min(index, seqLen)
        
        guard let sliced = try? MLMultiArray(shape: [1, NSNumber(value: newLen)], dataType: array.dataType) else {
            return nil
        }
        
        for i in 0..<newLen {
            sliced[[0, NSNumber(value: i)]] = array[[0, NSNumber(value: i)]]
        }
        
        return sliced
    }
    
    func getLogitsAtStep(_ logits: MLMultiArray, step i: Int) -> [Float] {
        
        var stepLogits = [Float](repeating: 0, count: vocabSize)
        
        for v in 0..<vocabSize {
            let index = i * vocabSize + v
            stepLogits[v] = logits[index].floatValue
        }
        return stepLogits
    }
    
    func softmax(_ values: [Float]) -> [Float] {
        let maxVal = values.max() ?? 0
        let exps = values.map { exp($0 - maxVal) }
        let sumExp = exps.reduce(0, +)
        
        return exps.map { $0 / sumExp}
    }
    
    func encode(tokens: [Int]) -> MLMultiArray? {
        let inputArray = makeMultiArrayInt(ints: tokens)
        print("Type of Input to Encoder: ", type(of: inputArray))
        describeMultiArray(inputArray)
        
        do {
            let input = try MLDictionaryFeatureProvider(dictionary: ["input_ids": inputArray])
            
            guard let encoderModel = encoderModel else {
                print("Encoder model not loaded")
                return nil
            }
            
            let encodeOutput = try encoderModel.prediction(from: input)
            let encoderHidden = encodeOutput.featureValue(for: "var_396")?.multiArrayValue
            return encoderHidden
        } catch {
            print("Encoding failed with error: \(error)")
            return nil
        }
    }
    
    func decode(hidden: MLMultiArray, decoder_inputs: MLMultiArray) -> MLMultiArray {
            
        do {
            let input = try MLDictionaryFeatureProvider(dictionary: ["decoder_input_ids": decoder_inputs, "encoder_hidden_states": hidden])
            
            let decoded = try decoderModel!.prediction(from: input)
            
            return decoded.featureValue(for: "var_829")!.multiArrayValue!
        } catch {
            print("Decoding failed with error: \(error)")
            return makeMultiArray(ints: [0])
        }
    }
    
    func autoRegression(inputText: String) -> String {
        var outputText: String = ""
        var Length: Int = 0
        
        //Tokenise the string.
        let inputTokens = tokeniser.encode(text: inputText)
        
        //Pass the inputTokens through the encoder, and pad them with zeros.
        if let encoded = encode(tokens: inputTokens) {
            let encoderHidden = convertToFloat32(padHidden(encoded, toLength: seqLen))

            //Create an array of size 50 with the output till now
            let outputTillNow = makeMultiArrayInt(ints: [Int](repeating: startTokenId, count: seqLen))
//            print("Decoder Inputs: ")
//            describeMultiArray(outputTillNow)
            for i in 0..<seqLen {
                Length = i+1
                let logits = decode(hidden: encoderHidden, decoder_inputs: outputTillNow)
//                print("Logits type : ", type(of: logits))
//                print("Logits : ", logits)
                let stepLogits = getLogitsAtStep(logits, step: i)
                let probabilities = softmax(stepLogits)
                if let index = probabilities.enumerated().max(by: {$0.element < $1.element})?.offset {
                    outputTillNow[[0, NSNumber(value: min(i+1, seqLen - 1))]] = NSNumber(value: index)
                    
                    if index == 0 {
                        break
                    }
                } else {
                    print("Error in autoregressive behaviour")
                }
                
            }
            let outputTokenIds = mlMultiArrayToIntArray(outputTillNow)
            outputText = tokeniser.decode(tokenIds: Array(outputTokenIds.prefix(Length).dropFirst()))
//            print(outputText)
        } else {
            print("Error in encoding and padding the input tokens.")
        }
        return outputText
    }
    
    func describeMultiArray(_ arr: MLMultiArray, maxElements: Int = 20) {
        let shape = arr.shape.map { $0.intValue }
        print("Shape: \(shape), DataType: \(arr.dataType)")
        
        var values = [Float]()
        let count = min(arr.count, maxElements)
        for i in 0..<count {
            values.append(arr[i].floatValue)
        }
        
        print("Values (first \(count)): ", values)
    }
}
