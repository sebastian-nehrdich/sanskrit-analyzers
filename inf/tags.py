def read_skt_tags(path):
    result = {}
    lines = open(path).readlines()
    lines = [line.split("\t") for line in lines]
    for line in lines:
        result[line[0]] = line[1].strip()
    return result

sanskrit_tags = read_skt_tags("data/sanskrit_tags.tsv")

def postprocess_sentence(sentence, mode="lemma-morphosyntax"):    
    """
    Postprocess the output of the model to a conllu-compatible format.
    """
    if mode == "lemma-morphosyntax" or mode == "segmentation-lemma-morphosyntax":        
        result = ""                
        for item in sentence.split(" "):
            if mode == "segmentation-lemma-morphosyntax":
                if len(item.split("_")) == 3:
                    unsandhied, lemma, short_tag = item.split("_")
                    if short_tag in sanskrit_tags:
                        short_tag = sanskrit_tags[short_tag]
                        if "Cpd" in short_tag:
                            unsandhied = unsandhied + "-"
                    result += f"{unsandhied}_{lemma}_{short_tag} "
            else:
                if len(item.split("_")) == 2:
                    lemma, short_tag = item.split("_")
                    if short_tag in sanskrit_tags:
                        short_tag = sanskrit_tags[short_tag]
                    result += f"{lemma}_{short_tag} "        
        return result
    elif mode == "lemma" or mode == "segmentation":
        return sentence.replace("_", " ")
    
