import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm.auto import tqdm
import pandas as pd


RANDOM_SEED = 42


class Tester(object):

    def __init__(self, task, model_service):
        self.task = task
        self.model_service = model_service
        self.test_dataset = self.task.get_tensor_dataset()

        self.device = self.model_service.device
        #print('cuda selection...')
        #if torch.cuda.is_available():    
        #    self.device = torch.device('cuda')
        #    print('There are %d GPU(s) available.' % torch.cuda.device_count())
        #    print('We will use the GPU:', torch.cuda.get_device_name(0))
        #else:
        #    print('No GPU available, using the CPU instead.')
        #    self.device = torch.device('cpu')
        #pd.set_option('precision', 2)

        self.config = self.model_service.get_config()
        self.tokenizer = self.model_service.get_tokenizer()
        
        self.model = self.model_service.get_model()
        self.model.to(self.device) # Runs the model on the GPU (if available)

        test_dataloader = self.make_dataloader(self.test_dataset)
        
        results_df = self.init_results_df(self.test_dataset)

        (results_df, df) = self.do_test(test_dataloader, results_df)

        results_df = self.add_detok_preds(results_df)
        results_df = self.remove_special_tokens(results_df)
        """
        try:
            results_df = self.add_detok_preds(results_df)
            results_df = self.remove_special_tokens(results_df)
            print('detokenized successfully!')
        except:
            print('detokenization issue!')
        """
        # results_df.to_pickle("results_df.pkl")
        # print('export completed successfully!')
        self.results_df = results_df

    def get_results(self):
        return self.results_df
        
    # @OK
    def init_results_df(self, test_dataset):
        
        test_df_index = [int(d[-1]) for d in test_dataset]
        test_df_columns = ['numeric_id', 'input_ids', 'input_mask', 'gold_labels', 'preds_1', 'sent']
        test_df = pd.DataFrame(index=test_df_index, columns=test_df_columns)
        
        for d in test_dataset:
            test_df.at[int(d[3]), 'input_ids'] = d[0].tolist()   #   [0]: input ids 
            test_df.at[int(d[3]), 'input_mask'] = d[1].tolist()  #   [1]: attention masks
            test_df.at[int(d[3]), 'gold_labels'] = d[2].tolist() #   [2]: labels
            test_df.at[int(d[3]), 'numeric_id'] = int(d[3])      #   [3]: numeric id
            test_df.at[int(d[3]), 'sent'] =  self.tokenizer.convert_ids_to_tokens(d[0].tolist())
            
        return test_df
        
    # @OK
    def make_dataloader(self, dataset):
        def _init_fn():
            np.random.seed(RANDOM_SEED)
        if dataset is None:
            dataloader = None
        else:
            dataloader = DataLoader(
                dataset,                         
                sampler = SequentialSampler(dataset),
                batch_size = self.config.batch_size,
                num_workers = 0,
                worker_init_fn=_init_fn )
        return dataloader
        
    def do_test(self, test_dataloader, results_df):
        
        training_stats = []
        all_epoch_preds = []

        epoch_predictions, avg_val_loss = self.test_model(test_dataloader, results_df, 1)
        all_epoch_preds.append(epoch_predictions)

        training_stats.append({
                'epoch': 1,
                'Training_Loss': -1,
                'Valid_Loss': avg_val_loss,
            })
        

        df_stats = pd.DataFrame(data=training_stats)  # training statistics
        df_stats = df_stats.set_index('epoch')        # Use the 'epoch' as the row index
        df_stats = df_stats[['Training_Loss','Valid_Loss']]

        return (results_df, df_stats)
    
    
    def test_model(self, test_dataloader, results_df, epoch):

        epoch_predictions = []
        total_eval_loss = 0
        
        self.model.eval()

        for batch in tqdm(test_dataloader, desc="predicting"):
            
            b_input_ids = batch[0].to(self.device)   #   [0]: input ids 
            b_input_mask = batch[1].to(self.device)  #   [1]: attention masks
            b_labels = batch[2].to(self.device)      #   [2]: labels (BILUO)
            b_numeric_ids = batch[3].to(self.device) #   [3]: numeric id

            with torch.no_grad():
                (loss, preds) = self.model( b_input_ids, 
                                             attention_mask = b_input_mask,
                                             labels = b_labels )

            total_eval_loss += loss.item()

            for i, pred_tensor in enumerate(preds):
                pred = pred_tensor.tolist()
                results_df.at[int(b_numeric_ids[i]), f'preds_{epoch}'] = pred
                epoch_predictions.append([pred, int(b_numeric_ids[i])])

        # LOG.info(f'test competed, compute loss')

        # Average loss on all batches
        avg_val_loss = total_eval_loss / len(test_dataloader)  if len(test_dataloader)>0 else total_eval_loss
        return epoch_predictions, avg_val_loss
    
    def remove_special_tokens(self, results_df):
        
        results_df["detok_preds_1_clean"] = None
        results_df["detok_sent_clean"] = None
        
        for idx, row in results_df.iterrows():
            sep_index = row.detok_sent.index(self.tokenizer.sep_token)
            row.at["detok_preds_1_clean"] = row.detok_preds_1[1:sep_index]
            row.at["detok_sent_clean"] = row.detok_sent[1:sep_index]
        
        return results_df


    def add_detok_preds(self, results_df):
        
        pred_cols = [c for c in results_df.columns if c.startswith('preds')]
        results_df['detok_gold_labels'] = ''
        results_df['detok_sent'] = ''
        results_df['detok_input_mask'] = ''
        
        for col in pred_cols:
            results_df['detok_'+col] = ''
        
        for _, line in results_df.iterrows():
            from_id_to_tok = self.tokenizer.convert_ids_to_tokens(line.input_ids)
            
            for col in pred_cols:
                detok_sent, detok_labels, detok_preds = self.mergeTokenAndPreserveData( from_id_to_tok,
                                                                                        line.gold_labels,
                                                                                        line[col] )
                line.at['detok_'+col] = detok_preds
            
            line['detok_gold_labels'] = detok_labels
            line['detok_sent'] = detok_sent
            line['detok_input_mask'] = [0 if x == self.tokenizer.pad_token else 1 for x in detok_sent]

        return results_df


    def mergeTokenAndPreserveData(self, sentence, labels, predictions):

        detok_sent = []
        detok_labels = []
        detok_predict = []

        for token, lab, pred in zip(sentence, labels, predictions):

            # CASE token to be added to the previous token
            if '##' in token:

                # rebuild the word
                detok_sent[-1] = detok_sent[-1] + token[2:]

                if pred > detok_predict[-1]:
                    detok_predict[-1] = pred
                #    LOG.info(' > Prediction updated')

            else:
                detok_sent.append(token)
                detok_labels.append(lab)
                detok_predict.append(pred)

        return detok_sent, detok_labels, detok_predict
