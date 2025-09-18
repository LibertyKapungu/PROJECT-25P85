Repo structure 
```bash
├── loizou_malab_code/             
│   ├── Wiener filtering/    
│   └── Spectral Subtraction/
│
├── src/                    
│   ├── dsp_algorithms/   
│   │   ├── __init__.py
│   │   ├── wiener_as.py
│   │   └── specsub.py
│   │
│   ├── metrics/              
│   │   ├── __init__.py
│   │   ├── objective_metrics.py 
│   │   └── MOS_metrics.py        
│   │
│   ├── machine_learning/             
│   │   ├── __init__.py
│   │   └── tensor_train.py
│   │
│   └── experiments/           
│       ├── EXP0/
│       │   ├── run matlab algs.m      
│       │   └── run python algs.m
│       ├── EXP1/
│       ├── EXP2/
│       ├── EXP3/
│       ├── EXP4/
│       └── EXP5/
│
├── sound_data/                      
│   ├── raw/                   
│   │   └── EARS/ ......
│   ├── EXP0/
│   ├── EXP1/
│   ├── EXP2-5/
│   └── processed/            
│       ├── spectral_processed_output/
│       │   ├── EXP0_output/
│       │   ├── EXP1_output/
│       │   ├── EXP2_output/
│       │   ├── EXP3_output/
│       │   ├── EXP4_output/
│       │   └── EXP5_output/
│       └── wiener_processed_output/
│           ├── EXP0_output/
│           ├── EXP1_output/
│           ├── EXP2_output/
│           ├── EXP3_output/
│           ├── EXP4_output/
│           └── EXP5_output/
│
├── models/                    
│   ├── pretrained/            
│   │   └── (tensor_train model files)
│   │
│   └── fine_tuned/            
│       └── tensor_train_v1/
│           ├── model.pt
│           └── training_log.json
│
├── results/                   
│   ├── EXP0/
│   ├── EXP1/.....
│   │
│   └── fine_tuning/           
│       └── hyperparameter_tuning_logs.txt
│
├── .gitignore                 
├── requirements.txt           
├── Commit_Msg_Stds.txt        
└── README.md         
```