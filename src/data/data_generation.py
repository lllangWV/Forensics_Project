import os
import sys
import shutil
import re
import random
import multiprocessing
import itertools
from dataclasses import dataclass
from functools import partial

import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps

from src.exceptions import CustomException
from src.logger import logging

Image.MAX_IMAGE_PIXELS = 933120000
SCRATCH_DIR = f"{os.sep}users{os.sep}lllang{os.sep}SCRATCH"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATASET_DIR = f"{PROJECT_DIR}{os.sep}datasets"

@dataclass
class DataGenerationConfig:
    """
    DataGenerationConfig is a dataclass that holds the configuration for data generation.
    """

    generated_data_dir: str= f"{DATASET_DIR}{os.sep}generated"

    split_images_dir: str= f"{DATASET_DIR}{os.sep}raw{os.sep}split_images{os.sep}default"
    match_nonmatch_file: str = f"{DATASET_DIR}{os.sep}raw{os.sep}match-nonmatch.xlsx"

    def __init__(self,from_scratch: bool=False):
        if from_scratch:
            if os.path.exists(self.generated_data_dir):
                shutil.rmtree(self.generated_data_dir)
            os.makedirs(self.generated_data_dir)

def get_random_permutation_without_replacement(permuation_list ):
    rand_i = random.randint(0,len(permuation_list)-1)
    random_tape_permutation = permuation_list.pop(rand_i)
    return random_tape_permutation

class DataGeneration:
    def __init__(self,from_scratch: bool=False):
        self.config = DataGenerationConfig(from_scratch=from_scratch)
    
    def initiate_data_generation(self,generated_dir, match_nonmatch_ratio=0.3):
        logging.info("Initiating data genetation")
        self.generated_dir = f"{self.config.generated_data_dir}{os.sep}{generated_dir}"
        if os.path.exists(self.generated_dir):
            shutil.rmtree(self.generated_dir)
        os.makedirs(self.generated_dir)

        try:
            df_train, df_test = self.generate_train_test(match_nonmatch_ratio=match_nonmatch_ratio )
            self.save_generated_data(df_train, split="train")
            self.save_generated_data(df_test, split="test")
        except Exception as e:
            raise CustomException(e,sys)

    def generate_train_test(self, match_nonmatch_ratio=0.3, test_size=0.2):
        
        df = pd.read_excel(self.config.match_nonmatch_file).drop(columns=['Unnamed: 0','flip_f', 'flip_b'])

        df['tape_f_1'] = df['tape_f1'] + '_' + df['side_f1']
        df['tape_f_2'] = df['tape_f2'] + '_' + df['side_f2']

        df['tape_b_1'] = df['tape_b1'] + '_' + df['side_b1']
        df['tape_b_2'] = df['tape_b2'] + '_' + df['side_b2']

        df = df[['tape_f_1','tape_f_2','tape_b_1','tape_b_2','match']]
        df['quality'] = df['tape_f_1'].apply(lambda x: x.split('_')[0])

        df_nonmatch = df[ df['match'] == 0]
        df_match = df[ df['match'] == 1]

        # X = df_match.drop('match', axis=1)
        # y = df_match[['match']]
        X = df_match

        df_train, df_test = train_test_split(df_match, stratify=df_match[ ['quality'] ], test_size=test_size,random_state=0)

        df_train = self.generate_nonmatches(df_train, match_nonmatch_ratio=match_nonmatch_ratio)
        df_test = self.generate_nonmatches(df_test, match_nonmatch_ratio=match_nonmatch_ratio)

        logging.info('Number of Matches : {0}'.format( df_match.shape[0] ))
        logging.info('Number of NonMatches : {0}'.format( df_nonmatch.shape[0] ))

        logging.info('Number of Train Nonmatches : {0}'.format( df_train[ df_train['match'] == 0].shape[0] ))
        logging.info('Number of Train Matches : {0}'.format( df_train[ df_train['match'] == 1].shape[0] ))

        logging.info('Number of Test Nonmatches : {0}'.format( df_test[ df_test['match'] == 0].shape[0] ))
        logging.info('Number of Test Matches : {0}'.format(df_test[ df_test['match'] == 1].shape[0] ))
        
        for i,x in enumerate([df_train, df_test]):
            if i==0:
                split = 'train'
            else:
                split = 'test'
            logging.info('______________________________________________________________')
            logging.info("Logging tape type ratio for {split} split".format(split=split))
            logging.info('______________________________________________________________')
            for quality in df['quality'].unique():
                logging.info('{0} qualities : {1}'.format( quality, x[ x['quality'] == quality].shape[0]/len(x) ))
        return df_train, df_test

    def generate_nonmatches(self, df, match_nonmatch_ratio=0.3):
        # Finding all match names and there permutations
        match_pairs = [[row['tape_f_1'], row['tape_f_2'], row['tape_b_1'], row['tape_b_2'], int(row['match']), row['quality']] for i, row in df.iterrows()]
        reversed_match_pairs = [[row['tape_f_2'], row['tape_f_1'],row['tape_b_2'], row['tape_b_1'], int(row['match']), row['quality']] for i, row in df.iterrows()]
        match_pairs.extend(reversed_match_pairs)
        
        # Finding tape names per quality type
        tape_names = {}
        tape_f_b_mapping = {}
        for quality in df['quality'].unique():
            tmp_list = df[ df['quality'] == quality]['tape_f_1'].values.tolist()
            tmp_list.extend(df[ df['quality'] == quality]['tape_f_2'].values.tolist())
            tape_names[quality] = tmp_list

            tmp_list_2 = df[ df['quality'] == quality]['tape_b_1'].values.tolist()
            tmp_list_2.extend(df[ df['quality'] == quality]['tape_b_2'].values.tolist())

            for tape_f,tape_b in zip(tmp_list, tmp_list_2):
                tape_f_b_mapping[tape_f] = tape_b

        # Finding permutations for each tape type
        tape_names_permutations={}
        for quality in tape_names.keys():
            permutations =  list(itertools.permutations(tape_names[quality], 2))
            random.shuffle(permutations)
            tape_names_permutations[quality] = permutations
            

        # # non_match_names for not including double nonmatches
        non_match_pairs = [] 
        isListFull = False
        while isListFull == False:

            rand_quality = random.choice(list(tape_names_permutations.keys()))

            # random permutation without replacement makes sure pairs are unique
            random_tape_permutation = get_random_permutation_without_replacement(tape_names_permutations[rand_quality])
            
            tape_f_1 = random_tape_permutation[0]
            tape_f_2 = random_tape_permutation[1]
            tape_b_1 = tape_f_b_mapping[tape_f_1]
            tape_b_2 = tape_f_b_mapping[tape_f_2]

            # nonmatch is 0, match is 1
            match = int(0)
            random_match_pair = [ tape_f_1, tape_f_2, tape_b_1, tape_b_2, match, rand_quality]

            # Make sure random_match_pair permutation is not one of the match pairs
            if random_match_pair not in match_pairs:
                non_match_pairs.append( random_match_pair)

            try:
                if (len(match_pairs)/2)/len(non_match_pairs) <= match_nonmatch_ratio:
                    isListFull = True
            except ZeroDivisionError:
                raise CustomException(ZeroDivisionError , sys)


        final_list = df.values.tolist()
        final_list.extend(non_match_pairs)
  
        df = pd.DataFrame(final_list, columns=['tape_f_1','tape_f_2','tape_b_1','tape_b_2','match','quality'])
        return df

    def save_generated_data(self, df, split:str):

        split_dir = f'{self.generated_dir}{os.sep}{split}'
        split_match_dir = f'{split_dir}{os.sep}match'
        split_nonmatch_dir = f'{split_dir}{os.sep}nonmatch'
        
        split_match_front_dir = f'{split_match_dir}{os.sep}front'
        split_match_back_dir = f'{split_match_dir}{os.sep}back'

        split_nonmatch_front_dir = f'{split_nonmatch_dir}{os.sep}front'
        split_nonmatch_back_dir = f'{split_nonmatch_dir}{os.sep}back'

        # Creating directories
        if os.path.exists(split_nonmatch_front_dir):
            shutil.rmtree(split_nonmatch_front_dir)
        os.makedirs(split_nonmatch_front_dir)

        if os.path.exists(split_nonmatch_back_dir):
            shutil.rmtree(split_nonmatch_back_dir)
        os.makedirs(split_nonmatch_back_dir)

        if os.path.exists(split_match_front_dir):
            shutil.rmtree(split_match_front_dir)
        os.makedirs(split_match_front_dir)

        if os.path.exists(split_match_back_dir):
            shutil.rmtree(split_match_back_dir)
        os.makedirs(split_match_back_dir)

        for i,row in df.iterrows():
            image_f1_path = f'{self.config.split_images_dir}{os.sep}{row["tape_f_1"]}.jpg'
            image_f2_path = f'{self.config.split_images_dir}{os.sep}{row["tape_f_2"]}.jpg'

            image_b1_path = f'{self.config.split_images_dir}{os.sep}{row["tape_b_1"]}.jpg'
            image_b2_path = f'{self.config.split_images_dir}{os.sep}{row["tape_b_2"]}.jpg'

            
            if row['match'] == 1:
                name_f = f'{row["tape_f_1"]}-{row["tape_f_2"]}'
                img_f = self.concatenate_images(image_f1_path, image_f2_path)
                img_f.save(f'{split_match_front_dir}{os.sep}{name_f}.jpg')

                name_b = f'{row["tape_b_1"]}-{row["tape_b_2"]}'
                img_b = self.concatenate_images(image_b1_path, image_b2_path)
                img_b.save(f'{split_match_back_dir}{os.sep}{name_b}.jpg')

            elif row['match'] == 0:
                name_f = f'{row["tape_f_1"]}-{row["tape_f_2"]}'
                img_f = self.concatenate_images(image_f1_path, image_f2_path)
                img_f.save(f'{split_nonmatch_front_dir}{os.sep}{name_f}.jpg')

                name_b = f'{row["tape_b_1"]}-{row["tape_b_2"]}'
                img_b = self.concatenate_images(image_b1_path, image_b2_path)
                img_b.save(f'{split_nonmatch_back_dir}{os.sep}{name_b}.jpg')

    def concatenate_images(self, image_a_path, image_b_path):
        img_a = Image.open(image_a_path)
        img_b = Image.open(image_b_path)

        img_a = img_a.rotate(180)
        img_a = ImageOps.flip(img_a)

        # concatenate the images horizontally
        img = Image.new('L', (img_a.width + img_b.width, img_a.height))
        img.paste(im=img_a, box=(0, 0))
        img.paste(im=img_b, box=(img_a.width, 0))
        return img
    
if __name__=='__main__':

    ################################################
    # Starting data injestion
    #################################################
    obj=DataGeneration(from_scratch=False)
    obj.initiate_data_generation(generated_dir='match_nonmatch_ratio_0.3', match_nonmatch_ratio=0.3)