    def _remove_puncs_numbers_stop_words(self, tokens):
        """Remove punctuations in the words, words including numbers and words in the stop_words list.

        :param tokens: list of string
        :return: list of string with cleaned version
        """
        # TODO: Implement this method
        #punc = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        result = []
        size = len(tokens)
        i=0
        while i < size:
            s = len(tokens[i])
            j = 0
            flag = True
            while j < s:
                if tokens[i][j] in string.punctuation:
                    tokens[i] = tokens[i][:j] + tokens[i][j+1:]
                    s-=1
                    flag = False
                else:
                    flag = True
                if flag:
                    j+=1
            i+=1
                    
        result2 = []
        result2 = [x for x in tokens if x.isalpha()]
        result = [x for x in result2 if x not in self.stop_words]
        return result
