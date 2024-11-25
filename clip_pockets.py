import os
import struct
import molgrid
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import multiprocessing as mp
from functools import partial


# Parameters (hardcoded)
RADIUS = 10.0  # Clipping radius in Ångströms
NUM_WORKERS = os.cpu_count() - 1  # Number of multiprocessing workers
TYPES_FILE_DIR = './types'  # Directory containing the types file
TYPES_FILE = 'it2_tt_v1.3_10p20n_train0.types'  # The types file to read
DATA_DIR = './CrossDocked2020'  # Directory containing the actual data files
CACHE_FILE = 'receptor_ligand_pairs.pkl'  # Output cache file

names = molgrid.GninaIndexTyper().get_type_names()

def elem(t):
    '''Convert type index into element string'''
    name = names[t]
    if 'Hydrogen' in name:
        return 'H'
    elif 'Carbon' in name:
        return 'C'
    elif 'Nitrogen' in name:
        return 'N'
    elif 'Oxygen' in name:
        return 'O'
    elif 'Sulfur' in name:
        return 'S'
    elif 'Phosphorus' == name:
        return 'P'
    elif 'Fluorine' == name:
        return 'F'
    elif 'Chlorine' == name:
        return 'Cl'
    elif 'Bromine' == name:
        return 'Br'
    elif 'Iodine' == name:
        return 'I'
    elif 'Magnesium' == name:
        return 'Mg'
    elif 'Manganese' == name:
        return 'Mn'
    elif 'Zinc' == name:
        return 'Zn'
    elif 'Calcium' == name:
        return 'Ca'
    elif 'Iron' == name:
        return 'Fe'
    elif 'Boron' == name:
        return 'B'
    else:
        return None  # For unknown elements
        

def read_gninatypes(file_path):
    """
    Read a binary .gninatypes file and extract atom coordinates and type indices.
    """
    struct_fmt = 'fffi'  # Structure: float, float, float, int
    struct_len = struct.calcsize(struct_fmt)
    struct_unpack = struct.Struct(struct_fmt).unpack_from

    atoms = []
    with open(file_path, 'rb') as tfile:
        for chunk in iter(partial(tfile.read, struct_len), b''):
            x, y, z, t = struct_unpack(chunk)
            atoms.append((x, y, z, t))
    return atoms  # List of tuples: (x, y, z, t)
    

def save_as_pdb(file_path, atoms):
    """
    Save atom data to a PDB file.
    """
    with open(file_path, 'w') as f:
        for i, (x, y, z, t) in enumerate(atoms, start=1):
            element = elem(t)
            if element:  # Skip atoms with unsupported elements
                f.write(f"HETATM{i:5d}  {element:2}  MOL     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}\n")
        f.write("END\n")
        

def load_item(item):
    """
    Load receptor and ligand data from gninatypes files.
    """
    receptor_path = os.path.join(DATA_DIR, item['receptor_path'])
    ligand_path = os.path.join(DATA_DIR, item['ligand_path'])
    receptor_atoms = read_gninatypes(receptor_path)
    ligand_atoms = read_gninatypes(ligand_path)
    return receptor_atoms, ligand_atoms
    

def process_item(item, processed_pairs, pair_lock):
    """
    Process a single receptor-ligand pair:
    - Clip the receptor to a specified radius around the ligand pose.
    - Save the clipped receptor.
    - Return the RMSD, Vina score, clipped receptor path, and ligand path.
    """
    try:
        rmsd = item['rmsd']
        receptor_path = item['receptor_path']
        ligand_path = item['ligand_path']
        vina_score = item['vina_score']

        # Construct unique key for receptor-ligand pair
        pair_key = (receptor_path, ligand_path)

        # Construct full paths
        receptor_full_path = os.path.join(DATA_DIR, receptor_path)
        ligand_full_path = os.path.join(DATA_DIR, ligand_path)

        # Determine the clipped receptor filename and path
        receptor_dir = os.path.dirname(receptor_full_path)
        receptor_filename = os.path.basename(receptor_full_path)
        ligand_filename = os.path.basename(ligand_full_path)
        receptor_base, receptor_ext = os.path.splitext(receptor_filename)
        ligand_base, _ = os.path.splitext(ligand_filename)
        clipped_receptor_filename = f"{receptor_base}_clipped_{ligand_base}{receptor_ext}"
        clipped_receptor_path = os.path.join(receptor_dir, clipped_receptor_filename)

        with pair_lock:
            if pair_key in processed_pairs:
                clipped_receptor_rel_path = processed_pairs[pair_key]
            else:
                receptor_atoms, ligand_atoms = load_item(item)
                ligand_coords = np.array([(x, y, z) for x, y, z, t in ligand_atoms])

                clipped_receptor_atoms = []
                for x, y, z, t in receptor_atoms:
                    receptor_coord = np.array([x, y, z])
                    distances = np.linalg.norm(ligand_coords - receptor_coord, axis=1)
                    if np.any(distances <= RADIUS):
                        clipped_receptor_atoms.append((x, y, z, t))
                        
                # Skip if no atoms are within the clipping radius
                if not clipped_receptor_atoms:
                    print(f"Skipping receptor {receptor_path}: no atoms within {RADIUS} Å of ligand {ligand_path}")
                    return None

                os.makedirs(receptor_dir, exist_ok=True)
                save_as_pdb(clipped_receptor_path, clipped_receptor_atoms)

                clipped_receptor_rel_path = os.path.relpath(clipped_receptor_path, DATA_DIR)
                processed_pairs[pair_key] = clipped_receptor_rel_path

        ligand_rel_path = ligand_path

        return {
            'rmsd': rmsd,
            'vina_score': vina_score,
            'clipped_receptor_path': clipped_receptor_rel_path,
            'ligand_path': ligand_rel_path
        }
    except Exception as e:
        print(f'Exception occurred while processing {item}: {e}')
        return None
        

if __name__ == '__main__':
    # Read the types file and create a list of items to process
    data = []
    types_file_path = os.path.join(TYPES_FILE_DIR, TYPES_FILE)
    with open(types_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip() == '':
                continue
            parts = line.strip().split()
            try:
                if len(parts) < 6:
                    print(f"Skipping line {line_num}: insufficient data.")
                    continue  # Skip lines that don't have at least 6 parts
                label = int(parts[0])  # Not used in this script
                rmsd = float(parts[2])  # RMSD to crystal (column index 2)
                receptor_path = parts[3]
                ligand_path = parts[4]
                vina_score_str = parts[5]  # Access the Vina score at index 5
                vina_score = float(vina_score_str.lstrip('#'))  # Remove '#' and convert to float
                data.append({
                    'rmsd': rmsd,
                    'vina_score': vina_score,
                    'receptor_path': receptor_path,
                    'ligand_path': ligand_path
                
            })
            except Exception as e:
                print(f"Error parsing line {line_num}: {line.strip()} - {e}")
                continue

    # Initialize a manager for shared data between processes
    manager = mp.Manager()
    processed_pairs = manager.dict()
    pair_lock = manager.Lock()

    # Create a pool of workers
    pool = mp.Pool(NUM_WORKERS)
    process_func = partial(process_item, processed_pairs=processed_pairs, pair_lock=pair_lock)

    # Process items with a progress bar
    results = []
    for result in tqdm(pool.imap_unordered(process_func, data), total=len(data)):
        if result is not None:
            results.append(result)

    pool.close()
    pool.join()

    # Save the cache file with RMSD, Vina score, clipped receptor path, and ligand path
    cache_df = pd.DataFrame(results)

    # Reorder the columns
    cache_df = cache_df[['rmsd', 'vina_score', 'clipped_receptor_path', 'ligand_path']]

    # Save as a pickle file
    cache_df.to_pickle(CACHE_FILE)

    print(f'Done. Processed {len(results)} receptor-ligand pairs.')
