"""
Data Preprocessing and Matching for Bank Failure Prediction

This script merges datasets and creates matched datasets of alive and dead banks
for different time spans and variable sets.
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from dateutil.relativedelta import relativedelta
from typing import Dict, Tuple, List
import re


class BankDataPreprocessor:
    """Handles preprocessing and matching of bank failure data."""
    
    def __init__(self, panel_path: str, cbr_path: str):
        """
        Initialize preprocessor with data paths.
        
        Args:
            panel_path: Path to panel.dta file
            cbr_path: Path to cbrdataT.dta file
        """
        self.panel_path = panel_path
        self.cbr_path = cbr_path
        self.result = None
        
    def load_and_merge_data(self):
        """Load panel and CBR data, then merge them."""
        print("Loading panel data...")
        df = pd.read_stata(self.panel_path)
        df = df.drop(columns="dt")
        df = df.rename(columns={'mt': 'dt'})
        df = df[["regn", "dt", "revok"]]
        
        print("Loading CBR data...")
        df_1 = pd.read_stata(self.cbr_path)
        
        # Create columns checking for missing data
        df_1['naA'] = df_1[['regn', 'dt'] + [col for col in df_1.columns if col.startswith('A')]].notna().sum(axis=1) == 2
        df_1['naF'] = df_1[['regn', 'dt'] + [col for col in df_1.columns if col.startswith('F')]] \
            .assign(month=lambda x: pd.to_datetime(x['dt']).dt.month) \
            .apply(lambda row: row.notna().sum() == 2 if row['month'] in [1, 4, 7, 10] else np.nan, axis=1)
        
        print("Decumulating profit and loss statement values...")
        df_1 = self._decumulate_pl_statement(df_1)
        
        print("Computing intermediate variables...")
        df_1 = self._compute_intermediate_variables(df_1)
        
        print("Computing CAMEL and Extended CAMEL variables...")
        df_1 = self._compute_camel_variables(df_1)
        
        # Select Extended CAMEL variables
        df_test = df_1[[
            "regn", "dt", "loga", "capa", "npllns", "pfta", "liqa", "canw", "nwa", "nwaa", "kata",
            "gb", "gsa", "ngs", "ngsa", "tsa", "lni", "lnia", "ila", "lha", "tlla", "ovl", "ovf",
            "dpc", "hda", "fda", "res", "reslni", "plla", "niia", "nltl", "cfb", "iira", "iiea",
            "tiea", "tira", "iiriiea"
        ]]
        
        # Merge with panel data
        print("Merging datasets...")
        result = pd.merge(df_test, df, how="left", on=["regn", "dt"])
        result['revok'] = pd.to_datetime(result['revok'])
        result['dt'] = pd.to_datetime(result['dt'])
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        self.result = result
        return result
    
    def _decumulate_pl_statement(self, df_1: pd.DataFrame) -> pd.DataFrame:
        """Decumulate profit and loss statement values."""
        df_1['month'] = df_1['dt'].dt.month
        
        F_cols = [col for col in df_1.columns if col.startswith('F')]
        df_1[F_cols] = df_1[F_cols] / 3
        
        for col in F_cols:
            df_1[col] = np.where(
                df_1['month'].isin([1, 7, 10]),
                df_1[col] - df_1[col].shift(3),
                np.where(df_1['month'] == 4, df_1[col], np.nan)
            )
        
        df_1['A7elpc_______'] = np.where(
            df_1['month'] == 2,
            df_1['A7elpc_______'],
            df_1['A7elpc_______'] - df_1['A7elpc_______'].shift(1)
        )
        
        n_ifax_cols = [col for col in df_1.columns if col.startswith('n') and col.endswith('_ifax')]
        for col in n_ifax_cols:
            df_1[col] = np.where(
                (df_1['month'].isin([1, 4, 7, 10])) & (df_1[col].isna()),
                -1,
                df_1[col]
            )
        
        df_1[F_cols + n_ifax_cols] = df_1[F_cols + n_ifax_cols].fillna(method='ffill')
        
        if 'naF' in df_1.columns:
            df_1['naF'] = df_1['naF'].fillna(method='ffill')
        
        for col in n_ifax_cols:
            df_1[col] = df_1[col].replace(-1, np.nan)
        
        return df_1
    
    def _compute_intermediate_variables(self, df_1: pd.DataFrame) -> pd.DataFrame:
        """Compute intermediate variables for CAMEL."""
        selected_columns_ass = [col for col in df_1.columns if col.startswith('A') and 'a' in col[:4]]
        selected_columns_cap = [col for col in df_1.columns if col.startswith('A7el')]
        selected_columns_liq = [col for col in df_1.columns if re.match(r'^A(1c|2a)a', col)]
        selected_columns_lns = [col for col in df_1.columns if re.match(r'^A4la__[bfgh]', col)]
        selected_columns_npl = [col for col in df_1.columns if re.match(r'^A4la__[bfgh].*odue', col)]
        
        df_1['ass'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_ass].sum(axis=1, skipna=True))
        df_1['cap'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_cap].sum(axis=1, skipna=True))
        df_1['liq'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_liq].sum(axis=1, skipna=True))
        df_1['lns'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_lns].sum(axis=1, skipna=True))
        df_1['npl'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_npl].sum(axis=1, skipna=True))
        df_1['pft'] = df_1['A7elpc_______']
        
        # Extended CAMELS intermediate variables
        df_1['a12'] = df_1['liq']
        
        selected_columns_a4 = [col for col in df_1.columns if re.match(r'^A4la', col)]
        df_1['a4'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_a4].sum(axis=1, skipna=True))
        
        selected_columns_a567 = [col for col in df_1.columns if re.match(r'^A(5n|6b|7e)a', col)]
        df_1['a567'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_a567].sum(axis=1, skipna=True))
        
        selected_columns_a89 = [col for col in df_1.columns if re.match(r'^A(8p|9o)a', col)]
        df_1['a89'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_a89].sum(axis=1, skipna=True))
        
        selected_columns_lpe = [col for col in df_1.columns if re.match(r'^F..elp', col)]
        df_1['lpe'] = np.where(df_1['naF'], np.nan, df_1[selected_columns_lpe].sum(axis=1, skipna=True))
        
        selected_columns_lpr = [col for col in df_1.columns if re.match(r'^F..rlp', col)]
        df_1['lpr'] = np.where(df_1['naF'], np.nan, df_1[selected_columns_lpr].sum(axis=1, skipna=True))
        
        selected_columns_iie = [col for col in df_1.columns if re.match(r'^F(2ae__...|3de__...|5ne_____|6be..___)____', col)]
        df_1['iie'] = np.where(df_1['naF'], np.nan, df_1[selected_columns_iie].sum(axis=1, skipna=True))
        
        selected_columns_iir = [col for col in df_1.columns if re.match(r'^F(2a|4l|5n|6b)r__...____$', col)]
        df_1['iir'] = np.where(df_1['naF'], np.nan, df_1[selected_columns_iir].sum(axis=1, skipna=True))
        
        selected_columns_tir = [col for col in df_1.columns if re.match(r'^F..r', col)]
        df_1['tir'] = np.where(df_1['naF'], np.nan, df_1[selected_columns_tir].sum(axis=1, skipna=True))
        
        selected_columns_tie = [col for col in df_1.columns if re.match(r'^F..e', col)]
        df_1['tie'] = np.where(df_1['naF'], np.nan, df_1[selected_columns_tie].sum(axis=1, skipna=True))
        
        selected_columns_liab = [col for col in df_1.columns if re.match(r'^A..l', col)]
        df_1['liab'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_liab].sum(axis=1, skipna=True))
        
        selected_columns_gs = [col for col in df_1.columns if re.match(r'^A(5n|6b)a..g', col)]
        df_1['gs'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_gs].sum(axis=1, skipna=True))
        
        return df_1
    
    def _compute_camel_variables(self, df_1: pd.DataFrame) -> pd.DataFrame:
        """Compute CAMEL and Extended CAMEL variables."""
        # CAMEL vars
        df_1['loga'] = np.log(df_1['ass'])
        df_1['capa'] = df_1['cap'] / df_1['ass']
        df_1['npllns'] = df_1['npl'] / df_1['lns']
        df_1['pfta'] = df_1['pft'] / df_1['ass']
        df_1['liqa'] = df_1['liq'] / df_1['ass']
        
        # Extended CAMEL vars (simplified - add full implementation as needed)
        df_1['canw'] = df_1['a89']
        
        selected_columns_nwa = [col for col in df_1.columns if re.match(r'^A9oa', col)]
        df_1['nwa'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_nwa].sum(axis=1, skipna=True))
        df_1['nwaa'] = df_1['nwa'] / df_1['ass']
        df_1['kata'] = np.where(
            df_1['naA'], np.nan,
            (df_1['a12']**2 + df_1['a4']**2 + df_1['a567']**2 + df_1['a89']**2) / df_1['ass']
        )
        
        selected_columns_gb = [col for col in df_1.columns if re.match(r'^A6ba..g', col)]
        df_1['gb'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_gb].sum(axis=1, skipna=True))
        
        selected_columns_ngs_base = [col for col in df_1.columns if re.match(r'^A(5n|6b)a', col)]
        df_1['gs'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_gb].sum(axis=1, skipna=True))
        df_1['gsa'] = df_1['gs'] / df_1['ass']
        df_1['ngs'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_ngs_base].sum(axis=1, skipna=True) - df_1['gs'])
        df_1['ngsa'] = df_1['ngs'] / df_1['ass']
        df_1['tsa'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_ngs_base].sum(axis=1, skipna=True) / df_1['ass'])
        
        selected_columns_lni = [col for col in df_1.columns if re.match(r'^A4la..[^bc]', col)]
        df_1['lni'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_lni].sum(axis=1, skipna=True))
        df_1['lnia'] = df_1['lni'] / df_1['ass']
        
        selected_columns_ila = [col for col in df_1.columns if re.match(r'^A4la..b', col)]
        df_1['ila'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_ila].sum(axis=1, skipna=True) / df_1['ass'])
        
        selected_columns_lha = [col for col in df_1.columns if re.match(r'^A4la..h', col)]
        df_1['lha'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_lha].sum(axis=1, skipna=True) / df_1['ass'])
        
        selected_columns_tlla = [col for col in df_1.columns if re.match(r'^A4la', col)]
        df_1['tlla'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_tlla].sum(axis=1, skipna=True) / df_1['ass'])
        
        selected_columns_ovl = [col for col in df_1.columns if re.match(r'^A4la.*odue$', col)]
        df_1['ovl'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_ovl].sum(axis=1, skipna=True))
        
        selected_columns_ovf = [col for col in df_1.columns if re.match(r'^A4la..f.*odue', col)]
        df_1['ovf'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_ovf].sum(axis=1, skipna=True) / df_1['ass'])
        
        selected_columns_dpc = [col for col in df_1.columns if re.match(r'A(2a|3d)l..[^bcg]', col)]
        df_1['dpc'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_dpc].sum(axis=1, skipna=True))
        
        selected_columns_hda = [col for col in df_1.columns if re.match(r'^A(2a|3d)l..h', col)]
        df_1['hda'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_hda].sum(axis=1, skipna=True) / df_1['ass'])
        
        selected_columns_fda = [col for col in df_1.columns if re.match(r'^A(2a|3d)l..f', col)]
        df_1['fda'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_fda].sum(axis=1, skipna=True) / df_1['ass'])
        
        selected_columns_res = [col for col in df_1.columns if re.match(r'^A(2a|4l|5n|6b|7e)c', col)]
        df_1['res'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_res].sum(axis=1, skipna=True))
        
        selected_columns_reslni_num = [col for col in df_1.columns if re.match(r'^A4lc..f', col)]
        selected_columns_reslni_den = [col for col in df_1.columns if re.match(r'^A4la..f', col)]
        df_1['reslni'] = np.where(
            df_1['naA'], np.nan,
            df_1[selected_columns_reslni_num].sum(axis=1, skipna=True) / df_1[selected_columns_reslni_den].sum(axis=1, skipna=True)
        )
        
        df_1['plla'] = (df_1['lpe'] - df_1['lpr']) / df_1['ass']
        df_1['niia'] = (df_1['iir'] - df_1['iie']) / df_1['ass']
        
        selected_columns_nltl_den = [col for col in df_1.columns if re.match(r'^A3dl', col)]
        df_1['nltl'] = np.where(
            df_1['naA'], np.nan,
            (df_1['liab'] - df_1[selected_columns_nltl_den].sum(axis=1, skipna=True)) / df_1['liab']
        )
        
        selected_columns_cfb = [col for col in df_1.columns if re.match(r'^A3dl..b', col)]
        df_1['cfb'] = np.where(df_1['naA'], np.nan, df_1[selected_columns_cfb].sum(axis=1, skipna=True))
        
        df_1['iira'] = df_1['iir'] / df_1['ass']
        df_1['iiea'] = df_1['iie'] / df_1['ass']
        df_1['tira'] = df_1['tir'] / df_1['ass']
        df_1['tiea'] = df_1['tie'] / df_1['ass']
        df_1['iiriiea'] = (df_1['iir'] + df_1['iie']) / df_1['ass']
        
        return df_1
    
    def get_variable_sets(self, df_1: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create different variable sets for modeling."""
        selected_columns_bs = [col for col in df_1.columns if (col.startswith('A') and 'a' in col[:4]) or
                              (col.startswith('A') and 'l' in col[:4]) or (col.startswith('A') and 'c' in col[:4])]
        selected_columns_re = [col for col in df_1.columns if (col.startswith('A') and 'l' in col[:4]) or
                              (col.startswith('A') and 'c' in col[:4])]
        selected_columns_pl = [col for col in df_1.columns if (col.startswith('F') and 'r' in col[:4]) or
                              (col.startswith('F') and 'e' in col[:4])]
        selected_columns_ass = [col for col in df_1.columns if col.startswith('A') and 'a' in col[:4]]
        
        return {
            "bs_pl": df_1[['regn', 'dt'] + selected_columns_bs + selected_columns_pl],
            "pl": df_1[['regn', 'dt'] + selected_columns_pl],
            "bs": df_1[['regn', 'dt'] + selected_columns_bs],
            "re": df_1[['regn', 'dt'] + selected_columns_re],
            "at": df_1[['regn', 'dt'] + selected_columns_ass]
        }


class BankMatcher:
    """Handles matching of dead and alive banks for model training."""
    
    def __init__(self, result: pd.DataFrame):
        """
        Initialize matcher with preprocessed data.
        
        Args:
            result: Merged and preprocessed DataFrame
        """
        self.result = result
    
    def get_dead_banks(self, T: int) -> pd.DataFrame:
        """
        Collect pool of dead banks with 2*T months of data.
        
        Args:
            T: Number of months (data will be 2*T months)
            
        Returns:
            DataFrame of dead banks with complete data
        """
        dead_id = self.result[~self.result['revok'].isna()]['regn'].unique()
        
        filtered_data = self.result[
            (self.result['regn'].isin(dead_id)) &
            (self.result['dt'] > self.result['revok'] - pd.DateOffset(months=2*T)) &
            (self.result.drop(columns=['revok']).notna().all(axis=1))
        ]
        
        filtered_data = filtered_data.sort_values(by=['regn', 'dt'], ascending=[True, False])
        filtered_data['n'] = filtered_data.groupby('regn').cumcount() + 1
        filtered_data = filtered_data.groupby('regn').filter(lambda x: len(x) >= 2*T)
        
        filtered_data['revok1'] = filtered_data['revok'].apply(lambda x: pd.Timestamp(x).replace(day=1))
        filtered_data['status'] = 0
        
        return filtered_data
    
    def find_alive_banks(self, date: pd.Timestamp, T: int) -> np.ndarray:
        """
        Find alive banks for a given failure date.
        
        Args:
            date: Date of bank failure
            T: Number of months
            
        Returns:
            Array of bank IDs that were alive
        """
        filtered_data = self.result[
            ((self.result['revok'].isna()) | (self.result['revok'] >= date + relativedelta(years=2))) &
            (self.result['dt'] > date - relativedelta(months=2*T)) &
            (self.result['dt'] <= date) &
            (self.result.drop(columns=['revok']).notna().all(axis=1))
        ]
        
        filtered_data = filtered_data.groupby('regn').filter(lambda x: len(x) >= 2*T)
        return filtered_data['regn'].unique()
    
    def expand_alive_data(self, date: pd.Timestamp, alive_regn: np.ndarray, T: int) -> pd.DataFrame:
        """
        Collect 2*T months of data for alive bank IDs.
        
        Args:
            date: Date of bank failure
            alive_regn: Array of alive bank IDs
            T: Number of months
            
        Returns:
            DataFrame with alive bank data
        """
        filtered_data = self.result[
            (self.result['regn'].isin(alive_regn)) &
            (self.result['dt'] > date - relativedelta(months=2*T)) &
            (self.result['dt'] <= date) &
            (self.result.drop(columns=['revok']).notna().all(axis=1))
        ]
        
        filtered_data = filtered_data.sort_values(by=['regn', 'dt'], ascending=[True, False])
        filtered_data['n'] = filtered_data.groupby('regn').cumcount() + 1
        filtered_data['status'] = 1
        
        return filtered_data
    
    def create_matched_dataset(
        self,
        variable_frame: pd.DataFrame,
        dead_id: int,
        T: int,
        max_attempts: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Match a dead bank to a comparable alive bank.
        
        Args:
            variable_frame: DataFrame with selected variables
            dead_id: ID of dead bank to match
            T: Number of months
            max_attempts: Maximum matching attempts
            
        Returns:
            Tuple of (dead_bank_array, alive_bank_array) or (None, None) if failed
        """
        dead_banks = self.get_dead_banks(T)
        
        for attempt in range(max_attempts):
            try:
                random = dead_banks[dead_banks['regn'] == dead_id].iloc[0]
                end_date = random.revok
                
                print(f"Attempt {attempt + 1}: End Date for Dead Bank {dead_id}: {end_date}")
                
                alive_regn = self.find_alive_banks(end_date, T)
                expanded_data = self.expand_alive_data(end_date, alive_regn, T)
                
                if expanded_data.empty:
                    print(f"Attempt {attempt + 1}: No alive banks found")
                    continue
                
                grouped_data = expanded_data.groupby('regn')
                sampled_regn = grouped_data['regn'].apply(lambda x: x.iloc[0]).sample(1).values
                
                if not sampled_regn.any():
                    continue
                
                # Process alive bank data
                sampled_data_expanded = expanded_data[expanded_data['regn'].isin(sampled_regn)]
                sampled_data_expanded = variable_frame.merge(
                    sampled_data_expanded[['regn', 'dt', 'n', 'status']],
                    on=['regn', 'dt'],
                    how='inner'
                )
                sampled_data_expanded = sampled_data_expanded.drop(columns=["dt", "status", "n", "regn"])
                sampled_data_expanded = sampled_data_expanded.fillna(0)
                
                if sampled_data_expanded.shape[0] != 2*T:
                    continue
                
                sampled_data_expanded = sampled_data_expanded.div(sampled_data_expanded.sum(axis=1), axis=0)
                sampled_data_expanded = sampled_data_expanded.to_numpy()
                
                # Process dead bank data
                sampled_data_dead = dead_banks[dead_banks['regn'] == dead_id]
                sampled_data_dead = variable_frame.merge(
                    sampled_data_dead[['regn', 'dt', 'n', 'status']],
                    on=['regn', 'dt'],
                    how='inner'
                )
                sampled_data_dead = sampled_data_dead.drop(columns=["dt", "status", "n", "regn"])
                sampled_data_dead = sampled_data_dead.fillna(0)
                
                if sampled_data_dead.shape[0] != 2*T:
                    return None, None
                
                sampled_data_dead = sampled_data_dead.div(sampled_data_dead.sum(axis=1), axis=0)
                sampled_data_dead = sampled_data_dead.to_numpy()
                
                return sampled_data_dead, sampled_data_expanded
                
            except (ValueError, IndexError) as e:
                print(f"Attempt {attempt + 1}: Error - {e}")
                continue
        
        print(f"Maximum attempts reached for dead bank {dead_id}")
        return None, None
    
    def create_full_dataset(
        self,
        variable_frame: pd.DataFrame,
        T: int
    ) -> Tuple[Dict, Dict]:
        """
        Create matched datasets for all dead banks.
        
        Args:
            variable_frame: DataFrame with selected variables
            T: Number of months
            
        Returns:
            Tuple of (dead_dict, alive_dict)
        """
        dead_banks = self.get_dead_banks(T)
        ids = dead_banks['regn'].unique()
        
        dead_dict = {}
        alive_dict = {}
        
        for i, bank_id in enumerate(tqdm(ids, desc="Matching banks")):
            sampled_data_dead, sampled_data_expanded = self.create_matched_dataset(
                variable_frame, bank_id, T, max_attempts=10
            )
            
            if sampled_data_dead is not None and sampled_data_expanded is not None:
                dead_dict[i+1] = sampled_data_dead
                alive_dict[i+1] = sampled_data_expanded
            else:
                print(f"Skipping bank {bank_id} due to insufficient data")
        
        return dead_dict, alive_dict


def save_datasets(
    T: int,
    variable_sets: Dict[str, pd.DataFrame],
    matcher: BankMatcher,
    output_dir: str
):
    """
    Save matched datasets for all variable sets.
    
    Args:
        T: Number of months (data will be 2*T)
        variable_sets: Dictionary of variable set DataFrames
        matcher: BankMatcher instance
        output_dir: Output directory path
    """
    folder_name = os.path.join(output_dir, f"Banks_{int(T * 2)}")
    os.makedirs(folder_name, exist_ok=True)
    
    for name, var_frame in variable_sets.items():
        print(f"\nProcessing variable set: {name}")
        dead_dict, alive_dict = matcher.create_full_dataset(var_frame, T)
        
        dead_path = os.path.join(folder_name, f"dead_bank_{name}_{int(T * 2)}.pickle")
        alive_path = os.path.join(folder_name, f"alive_bank_{name}_{int(T * 2)}.pickle")
        
        with open(dead_path, 'wb') as f:
            pickle.dump(dead_dict, f, protocol=4)
        
        with open(alive_path, 'wb') as f:
            pickle.dump(alive_dict, f, protocol=4)
        
        print(f"Saved {len(dead_dict)} matched pairs for {name}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Preprocess and match bank failure data'
    )
    parser.add_argument(
        '--panel-path',
        type=str,
        required=True,
        help='Path to panel.dta file'
    )
    parser.add_argument(
        '--cbr-path',
        type=str,
        required=True,
        help='Path to cbrdataT.dta file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Output directory for matched datasets'
    )
    parser.add_argument(
        '--time-span',
        type=int,
        default=6,
        help='Time span T (data will be 2*T months)'
    )
    parser.add_argument(
        '--variable-sets',
        nargs='+',
        choices=['bs_pl', 'bs', 'pl', 're', 'at', 'all'],
        default=['all'],
        help='Variable sets to process'
    )
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = BankDataPreprocessor(args.panel_path, args.cbr_path)
    
    # Load and merge data
    result = preprocessor.load_and_merge_data()
    
    # Get variable sets
    # Reload CBR data to get variable sets
    df_1 = pd.read_stata(args.cbr_path)
    variable_sets = preprocessor.get_variable_sets(df_1)
    
    # Filter variable sets based on arguments
    if 'all' not in args.variable_sets:
        variable_sets = {k: v for k, v in variable_sets.items() if k in args.variable_sets}
    
    # Initialize matcher
    matcher = BankMatcher(result)
    
    # Save datasets
    save_datasets(args.time_span, variable_sets, matcher, args.output_dir)
    
    print(f"\nPreprocessing complete! Datasets saved to {args.output_dir}")


if __name__ == "__main__":
    main()