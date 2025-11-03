from predictor import DeepTYLCV_Predictor
from inference import Inferencer
import yaml

# Initialize predictor
config = yaml.safe_load(open('/home/vinoth/Hari_proj/TYLCV/webserver/Github_code/DeepTYLCV/DeepTYLCV_webserver_data/sweeper_59_part_2/config.yaml'))

predictor = DeepTYLCV_Predictor(
    model_config=config['model'], # A dict of model configuration, which can be easily loaded from yaml config file (`model_config` key).
    ckpt_dir='/path/to/model_files', # The path to all folds' checkpoints. 
    nfold=5, # Number of checkpoints inside `ckpt_dir`. Please use 5, which is the default of our training configuration.
    device='cuda', # Device, 'cpu' or 'cuda'.
)
infer =  Inferencer(predictor,scaler_path='path/to/scaler/model' device='cuda') # You can use device 'cpu' or 'cuda', please be consistent with device of `predictor`

# Predict FASTA file
# outputs = infer.predict_fasta_file(
#     fasta_file='/home/vinoth/Hari_proj/TYLCV/webserver/Github_code/data/TYLCV_gun_test.fasta', # The path to FASTA file, check FASTA format at: https://en.wikipedia.org/wiki/FASTA_format 
#     threshold=0.5, # Classification threshold
#     batch_size=4 # Size of each iteration of prediction. You can increase the batch size for faster speed processing if having enough computing resources.
# )

# Predict sequences
outputs_seq = infer.predict_sequences(
    data_dict={'Severe_C1_NC_004005.1_C1': 'MPRLFKIYAKNYFLTYPNCSLSKEEALSQLKNLETPTNKKYIKVCRELHENGEPHLHVLIQFEGKYQCKNQRFFDLVSPNRSAHFHPNIQAAKSSTDVKTYVEKDGDFIDFGVFQIDGRSARGGQQSANDAYAEALNSGNKSEALNILKEKAPKDYILQFHNLSSNLDRIFSPPLEVYVSPFLSSSFNQVPDELEEWVAENVVSSAARPWRPNSIVIEGDSRTGKTMWARSLGPHNYLCGHLDLSPKVYSNDAWYNVIDDVDPHYLKHFKEFMGAQRDWQSNTKYGKPIQIKGGIPTIFLCNPGPTSSYREYLDEEKNISLKNWALKNATFVTLYEPLFASINQGPTQDSQEETNKA'}, 
    threshold=0.5, 
    batch_size=4
)

# Save results to CSV file
# print(outputs_seq)
infer.save_csv_file(outputs_seq, './prediction_results_severity_fasta_file.csv')

# Save results to CSV file
# infer.save_csv_file(outputs, '/path/to/csv/file')
