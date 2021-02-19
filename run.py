import csv
import os
import sys

import lexicon_reader
import sentiment


def get_training_data(filedir):
    with open(os.path.join(filedir, 'train.csv')) as csvfile:
        training_data = [row for row in csv.DictReader(csvfile, delimiter=',')]
        for entry in training_data:
            with open(os.path.join(filedir, 'train', entry['FileIndex'] + '.txt')) as reviewfile:
                entry['Review'] = reviewfile.read()
    return training_data


def get_training_accuracy(data, swn_dict, inqtabs_dict):
    #, afinn_dict
    num_correct = 0
    etype_files = {}
    for etype in ["fp", "fn", "tp", "tn"]:
        etype_files[etype] = open(etype + '.txt', 'w')
    for row in data:
        sentiment_prediction = sentiment.classify(row['Review'],swn_dict,inqtabs_dict)
        #, afinn_dict
        sentiment_label = int(row['Category'])
        if sentiment_prediction == sentiment_label:
            num_correct += 1
        etype = sentiment.get_error_type(sentiment_prediction, sentiment_label)
        etype_files[etype].write("%s\t%s\n"%(row['FileIndex'], row['Review']))
    accuracy = num_correct * 1.0 / len(data)
    for etype in ["fp", "fn", "tp", "tn"]:
        etype_files[etype].close()
    print("Accuracy: " + str(accuracy))
    return accuracy


def write_predictions(filedir, inqtabs_dict, swn_dict, output_file_name):
    #, afinn_dict
    testfiledir = os.path.join(filedir, 'test')
    with open(output_file_name, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=',', fieldnames=['FileIndex', 'Category'])
        writer.writeheader()
        for filename in sorted(os.listdir(testfiledir), key=lambda x: int(os.path.splitext(x)[0])):
            with open(os.path.join(testfiledir, filename)) as reviewfile:
                review = reviewfile.read()
                prediction = dict()
                prediction['FileIndex'] = os.path.splitext(filename)[0]
                prediction['Category'] = sentiment.classify(review, swn_dict, inqtabs_dict)
                #,  afinn_dict
                writer.writerow(prediction)


if __name__ == "__main__":
    filedir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    output_file_name = sys.argv[2] if len(sys.argv) > 2 else 'test.csv'
    print("Reading data")
    data = get_training_data(filedir)
    lexicon_dir = os.path.join(filedir, 'lexicon')
    inqtabs_dict = lexicon_reader.read_inqtabs(os.path.join(lexicon_dir, 'inqtabs.txt'))
    # print(inqtabs_dict)
    swn_dict = lexicon_reader.read_senti_word_net(os.path.join(lexicon_dir, 'SentiWordNet_3.0.0_20130122.txt'))
    # print(swn_dict)
    
    
    
    #??????????????????????????????????????????????????
    #afinn_dict = lexicon_reader.read_afinn(os.path.join(lexicon_dir, 'afinn.txt'))
    #afinn_dict = dict(map(lambda (k,v): (k,int(v)), 
           #          [ line.split('\t') for line in open("afinn.txt")]))
    '''
    with open('/Users/zhuoer/Downloads/nlp-hw1/lexicon/afinn.txt') as f:
        afinn_dict = {}
        for line in f.readlines():
            if len(line) > 1:
                line = line.strip()
                k,v = line.split('\t')
                afinn_dict[k] = v
    '''
    
    print("Classifying")
    get_training_accuracy(data, inqtabs_dict, swn_dict)
    #, swn_dict,
    print("Writing output")
    write_predictions(filedir, inqtabs_dict, swn_dict, output_file_name)
    #swn_dict,
