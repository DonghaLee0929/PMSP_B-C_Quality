# config.py

experiments = [
    {
        'name': 'base_high',
        'mode': 'framework',
        'jobs': [30, 40, 50, 60, 70, 80],
        'machines': [2, 3, 4, 5],
        'seeds': range(10, 110, 10),
        'env_config': {'tight': 'base', 'quality': 'high'},
        'run_args': {
            'asn_max_time': 3600, 
            'seq_max_time': 180, 
            'max_time': 3600,
        }
    },
    {
        'name': 'low_high',
        'mode': 'framework',
        'jobs': [30, 40, 50, 60, 70, 80],
        'machines': [2, 3, 4, 5],
        'seeds': range(10, 110, 10),
        'env_config': {'tight': 'low', 'quality': 'high'},
        'run_args': {
            'asn_max_time': 3600, 
            'seq_max_time': 180, 
            'max_time': 3600,
        }
    },
    {
        'name': 'high_high',
        'mode': 'framework',
        'jobs': [30, 40, 50, 60, 70, 80],
        'machines': [2, 3, 4, 5],
        'seeds': range(10, 110, 10),
        'env_config': {'tight': 'high', 'quality': 'high'},
        'run_args': {
            'asn_max_time': 3600, 
            'seq_max_time': 180, 
            'max_time': 3600,
        }
    },
    {
        'name': 'cp_only',
        'mode': 'cp',
        'jobs': [30, 40, 50, 60, 70, 80], 
        'machines': [2, 3, 4, 5],
        'seeds': range(10, 110, 10), 
        'env_config': {'tight': 'base', 'quality': 'high'}, 
        'run_args': {'max_time': 3600}
    },
    {
        'name': 'no_guide',
        'mode': 'framework',
        'jobs': [30, 60],
        'machines': [2, 3, 4, 5],
        'seeds': range(10, 110, 10),
        'env_config': {'tight': 'base', 'quality': 'high'},
        'run_args': {
            'asn_max_time': 3600, 
            'seq_max_time': 180, 
            'max_time': 3600,
            'guide_generation_TF': False, 
        }
    },
    {
        'name': 'no_column',
        'mode': 'framework',
        'jobs': [30, 40, 50, 60, 70, 80],
        'machines': [2, 3, 4, 5],
        'seeds': range(10, 110, 10),
        'env_config': {'tight': 'base', 'quality': 'high'},
        'run_args': {
            'asn_max_time': 3600, 
            'seq_max_time': 180, 
            'max_time': 3600,
            'optimal_guide_TF': False,
        }
    },
    {
        'name': 'large_size',
        'mode': 'framework',
        'jobs': [100, 200],
        'machines': [20, 25],
        'seeds': range(110, 210, 10),
        'env_config': {'tight': 'base', 'quality': 'high'},
        'run_args': {
            'asn_max_time': 60, 
            'seq_max_time': 180, 
            'max_time': 3600,
            'initial_guide_TF': False,
        }
    },
    {
        'name': 'no_column_large',
        'mode': 'framework',
        'jobs': [100, 200],
        'machines': [20, 25],
        'seeds': range(10, 110, 10),
        'env_config': {'tight': 'base', 'quality': 'high'},
        'run_args': {
            'asn_max_time': 60, 
            'seq_max_time': 180, 
            'max_time': 3600,
            'optimal_guide_TF': False,
            'initial_guide_TF': False,
        }
    }
]