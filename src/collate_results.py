from nlb_tools.collate_results import read_concat_results
from pathlib import Path

if __name__=='__main__':
    variant = 'mc_maze_20'
    base_path = Path(f'/home/kabird/STNDT_fewshot/ray_results/{variant}_lite')
    df = read_concat_results(base_path,endswith='_results_all11.csv')
    # df.drop(columns=['index'],inplace=True)
    
    df.to_csv(str(base_path)+'_all11.csv')

    print(df['id'])
    
