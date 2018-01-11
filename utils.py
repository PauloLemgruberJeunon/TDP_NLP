def measurePOSTagAccuracy(tagged_sents, tagged_words):
    comp_tagged_sents = []
    total_count = 0
    right_count = 0
    print('')
    for i in range(0, len(tagged_sents)):
        print('[Accuracy] Current stage = ' + str(i))
        k = 0
        j = 0
        local_count = 0
        local_right_count = 0
        while j < len(tagged_sents[i]):
            total_count += 1
            local_count += 1
            if tagged_sents[i][j][1] == '-NONE-':
                while tagged_sents[i][j][1] == '-NONE-':
                    j+=1
                while tagged_sents[i][j][0] != tagged_words[i][k][0]:
                    k+=1
            if tagged_sents[i][j][1] == tagged_words[i][k][1]:
                right_count += 1
                local_right_count += 1

            k += 1
            j += 1
        print('Local iter accuracy = ' + str(local_right_count/local_count) + '\n')

    return right_count/total_count

#*******************************************************************************************