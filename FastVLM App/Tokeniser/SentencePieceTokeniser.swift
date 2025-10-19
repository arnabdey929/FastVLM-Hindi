import Foundation
import SentencepieceTokenizer

class Tokeniser {
    var idToToken : [Int: String] = [:]
    var tokenToId : [String: Int] = [:]
    var spmTokenizer: SentencepieceTokenizer?
    let spmSpace = "\u{2581}"
    
    init() {
        
        guard let modelPath = Bundle.main.path(forResource: "source", ofType: "spm") else {
            print("Could not find source.spm in bundle")
            return
        }
        
        guard let vocabURL = Bundle.main.url(forResource: "vocab_M", withExtension: "json") else {
            print("Error with vocab.json file")
            return
        }
        
        do {
            let spm = try SentencepieceTokenizer(modelPath: modelPath)
            self.spmTokenizer = spm
            
            let data = try Data(contentsOf: vocabURL)
            guard let vocab = try JSONSerialization.jsonObject(with: data) as? [String: Int] else {
                print("Error with reading vocab.json file")
                return
            }
            self.tokenToId = vocab
            
            self.idToToken = Dictionary(uniqueKeysWithValues: vocab.map { ($1, $0) })
            
        } catch {
            print("Error: ", error)
        }
    }
    
    func encode(text: String)->[Int] {
        var embeddings : [Int] = []
        
        do {
            let normalised_text = self.spmSpace + text.replacingOccurrences(of: " ", with: self.spmSpace)
            if let pieces = try self.spmTokenizer?.encode(normalised_text) {
                var tokens : [String] = []
                
                for piece in pieces {
                    if let token = try self.spmTokenizer?.idToToken(piece) {
                        tokens.append(token)
                    }
                }
                
                var ids = tokens.map { self.tokenToId[$0] ?? 0}
                ids.append(0)
                embeddings = ids
            } else {
                print("Warning : Tokeniser returned wrong values")
            }
            
            
        } catch {
            print("Error: ", error)
        }
        
        
        return embeddings
    }
    
    func decode(tokenIds: [Int])->String {
        let tokens = tokenIds.compactMap { self.idToToken[$0] }
        let text = tokens.joined(separator: "")
            .replacingOccurrences(of: self.spmSpace, with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return text
    }
}
