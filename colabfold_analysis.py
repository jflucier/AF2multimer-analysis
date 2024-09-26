#!/usr/bin/python3

from argparse import ArgumentParser
import itertools
import argparse
import glob
import multiprocessing as mp
import os
import re
import lzma
import gzip
import numpy as np
import math
from statistics import mean
import pandas as pd
import sys
import shutil
from collections import defaultdict



#dict for converting 3 letter amino acid code to 1 letter code
aa_3c_to_1c = {
    "ALA": 'A',
    "CYS": 'C',
    "ASP": 'D',
    "GLU": 'E',
    "PHE": 'F',
    "GLY": 'G',
    "HIS": 'H',
    "ILE": 'I',
    "LYS": 'K',
    "LEU": 'L',
    "MET": 'M',
    "ASN": 'N',
    "PRO": 'P',
    "GLN": 'Q',
    "ARG": 'R',
    "SER": 'S',
    "THR": 'T',
    "VAL": 'V',
    "TRP": 'W',
    "TYR": 'Y',
}


def join_csv_files(files: list, output_name: str, sort_col: str = None, sort_ascending: bool = False, headers=None):
    """
        Join multiple CSV files into a single file.

        :param files (list): A list of file paths to CSV files to be joined.
        :param output_name (str): The name of the output file.
        :param sort_col (str, optional): The column header of the final CSV column by which to sort the rows by.
        :param sort_ascending (bool, optional): The sort direction to use when sorting the final output CSV.
        :param headers (list, optional): A list of column names for the output file. If not provided, the column names from the first input file are used.
    """
    if (len(files) < 1):
        return

    all_dfs = []
    for f in files:
        all_dfs.append(pd.read_csv(f))

    combo_df = pd.concat(all_dfs, ignore_index=True)

    if headers is not None:
        combo_df.columns = headers

    if sort_col:
        combo_df.sort_values(by=[sort_col], ascending=sort_ascending, inplace=True)
    combo_df.to_csv(output_name, index=None)


def distribute(lst: list, n_bins: int) -> list:
    """
        Returns a list containg n_bins number of lists that contains the items passed in with the lst argument

        :param lst: list that contains that items to be distributed across n bins
        :param n_bins: number of bins/lists across which to distribute the items of lst
    """
    if n_bins < 1:
        raise ValueError('The number of bins must be greater than 0')

    #cannot have empty bins so max number of bin is always less than or equal to list length
    n_bins = min(n_bins, len(lst))
    distributed_lists = []
    for i in range(0, n_bins):
        distributed_lists.append([])

    for i, item in enumerate(lst):
        distributed_lists[i % n_bins].append(item)

    return distributed_lists


def get_af_model_num(filename) -> int:
    """
        Returns the Alphafold model number from an input filestring as an int

        :param filename: string representing the filename from which to extract the model number
    """

    if "model_" not in filename: return 0

    model_num = int(re.findall(r'model_\d+', filename)[0].replace("model_", ''))
    return model_num


def get_finished_complexes(path: str, search_str: str = '.done.txt') -> list:
    """
        Returns a list of string values representing the names of the complexes that were found in a specified path (folder)

        :param path: string representing the path/folder to search for complexes
        :param search_str: string representing the pattern to use when searching for the completed complexes. By default this is *.done.txt because Colabfold outputs 1 such file per complex.
    """

    done_files = glob.glob(os.path.join(path, '*' + search_str))
    complex_names = [os.path.basename(f).replace(search_str, '') for f in done_files]
    return complex_names


def get_filepaths_for_complex(path: str, complex_name: str, pattern: str = '*') -> list:
    """
        Helper methdof for returning a list of filepaths (strs) that match the specified GLOB pattern

        :param path: string representing the path/folder to search for complexes
        :param complex_name: string that represents the name of a complex
        :param pattern: string representing the pattern to use when searching for files belonging to complex. Ex: *.pdb, *.json, etc
    """

    glob_str = os.path.join(path, complex_name + pattern)
    return sorted(glob.glob(glob_str))


def get_pae_values_from_json_file(json_filename) -> list:
    """
        Returns a list of string values representing the pAE(predicated Aligned Error) values stored in the JSON output

        :param json_filename: string representing the JSON filename from which to extract the PAE values
    """

    if not os.path.isfile(json_filename):
        raise ValueError('Non existing PAE file was specified')

    scores_file = None
    if (json_filename.endswith('.xz')):
        scores_file = lzma.open(json_filename, 'rt')
    elif (json_filename.endswith('.gz')):
        scores_file = gzip.open(json_filename, 'rt')
    elif (json_filename.endswith('.json')):
        scores_file = open(json_filename, 'rt')
    else:
        raise ValueError('pAE file with invalid extension cannot be analyzed. Only valid JSON files can be analyzed.')

    #read pae file in as text
    file_text = scores_file.read()
    pae_index = file_text.find('"pae":')
    scores_file.close()

    #Transform string representing 2d array into a 1d array of strings (each string is 1 pAE value). We save time by not unecessarily converting them to numbers before we use them.
    pae_data = file_text[pae_index + 6:file_text.find(']]', pae_index) + 2].replace('[', '').replace(']', '').split(',')

    if len(pae_data) != int(math.sqrt(len(pae_data))) ** 2:
        #all valid pAE files consist of an N x N matrice of scores
        raise ValueError('pAE values could not be parsed from files')

    return pae_data


def dist2(v1, v2) -> float:
    """
        Returns the square of the Euclian distance between 2 vectors carrying 3 values representing positions in the X,Y,Z axis

        :param v1: a vector containing 3 numeric values represening X, Y, and Z coordinates
        :param v2: a vector containing 3 numeric values represening X, Y, and Z coordinates
    """

    if len(v1) != 3:
        raise ValueError('3D coordinates require 3 values')

    if len(v2) != 3:
        raise ValueError('3D coordinates require 3 values')

    return (v1[0] - v2[0]) ** 2 + (v1[1] - v2[1]) ** 2 + (v1[2] - v2[2]) ** 2


def atom_from_pdb_line(atom_line: str) -> dict:
    """
        Parses a single line string in the standard PDB format and returns a list a dict that represents an atom with 3d coordinates and a type(element)

        :param atom_line: string representing a single line entry in a PDB file which contains information about an atom in a protein structure
    """
    coordinates = np.array([float(atom_line[30:38]), float(atom_line[38:46]), float(atom_line[46:54])])
    return {"type": atom_line[13:16].strip(), "xyz": coordinates, }


def get_closest_atoms(res1: dict, res2: dict):
    """
        Find the two closest atoms between two residues and returns the minimum distance as well as the closest atoms from each residue.

        :param res1: A dictionary representing the first residue. It should have a key 'atoms' that contains a list of dictionaries, where each dictionary represents an atom with keys 'xyz' (a list of three floats representing the x, y, and z coordinates) and 'name' (a string representing the name of the atom).
        :param res2: A dictionary representing the second residue. It should have a key 'atoms' that contains a list of dictionaries, where each dictionary represents an atom with keys 'xyz' (a list of three floats representing the x, y, and z coordinates) and 'name' (a string representing the name of the atom).
    """
    min_d2 = 1e6
    atoms = [None, None]
    for a1 in res1['atoms']:
        for a2 in res2['atoms']:
            d2 = dist2(a1["xyz"], a2["xyz"])
            if d2 < min_d2:
                min_d2 = d2
                atoms[0] = a1
                atoms[1] = a2
    return (min_d2, atoms)


def get_lines_from_pdb_file(pdb_filename: str) -> list:
    """
        Returns the contents of a protein databank file (PDB) file as a list of strings

        :param pdb_filename: string representing the path of the PDB file to open and parse (can handle PDB files that have been compressed via GZIP or LZMA)
    """

    if not os.path.isfile(pdb_filename):
        raise ValueError('Non existing PDB file was specified')

    pdb_file = None
    if (pdb_filename.endswith('.xz')):
        pdb_file = lzma.open(pdb_filename, 'rt')
    elif (pdb_filename.endswith('.gz')):
        pdb_file = gzip.open(pdb_filename, 'rt')
    elif (pdb_filename.endswith('.pdb')):
        pdb_file = open(pdb_filename, 'rt')
    else:
        raise ValueError('Unable to parse a PDB file with invalid file extension')

    pdb_data = pdb_file.read()
    pdb_file.close()
    return pdb_data.splitlines()


#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#FROM ELOFSSON BLOCK (https://gitlab.com/ElofssonLab/FoldDock/-/blob/9a1a26ced4f6b8b9bc65a7ac76999118c292b80d/src/pdockq.py)

def parse_atm_record(line: str) -> dict:
    """
        Returns a dict of values associated with an ATOM entry in a PDB file. A helper function defined for get_pdockq_elofsson

        :param line: A string representing a single line from an ATOM entry
    """
    record = defaultdict()
    record['name'] = line[0:6].strip()
    record['atm_no'] = int(line[6:11])
    record['atm_name'] = line[12:16].strip()
    record['atm_alt'] = line[17]
    record['res_name'] = line[17:20].strip()
    record['chain'] = line[21]
    record['res_no'] = int(line[22:26])
    record['insert'] = line[26].strip()
    record['resid'] = line[22:29]
    record['x'] = float(line[30:38])
    record['y'] = float(line[38:46])
    record['z'] = float(line[46:54])
    record['occ'] = float(line[54:60])
    record['B'] = float(line[60:66])

    return record


def get_pdockq_elofsson(pdb_filepath: str, chains: list = None) -> float:
    """
        Returns the pdockQ score as defined by https://www.nature.com/articles/s41467-022-28865-w

        :param pdb_filepath: string representing the path of the PDB file to open and parse (can handle PDB files that have been compressed via GZIP or LZMA)
        :param chain: an optional list of the chains to be used for calculating the pDOCKQ score
    """

    chain_coords, chain_plddt = {}, {}
    for line in get_lines_from_pdb_file(pdb_filepath):
        if line[0:4] != 'ATOM':
            continue
        record = parse_atm_record(line)
        if chains and record['chain'] not in chains:
            continue
        #Get CB - CA for GLY
        if record['atm_name'] == 'CB' or (record['atm_name'] == 'CA' and record['res_name'] == 'GLY'):
            if record['chain'] in [*chain_coords.keys()]:
                chain_coords[record['chain']].append([record['x'], record['y'], record['z']])
                chain_plddt[record['chain']].append(record['B'])
            else:
                chain_coords[record['chain']] = [[record['x'], record['y'], record['z']]]
                chain_plddt[record['chain']] = [record['B']]

    #Convert to arrays
    for chain in chain_coords:
        chain_coords[chain] = np.array(chain_coords[chain])
        chain_plddt[chain] = np.array(chain_plddt[chain])

    #Get coords and plddt per chain
    ch1, ch2 = [*chain_coords.keys()]
    coords1, coords2 = chain_coords[ch1], chain_coords[ch2]
    plddt1, plddt2 = chain_plddt[ch1], chain_plddt[ch2]

    #Calc 2-norm
    mat = np.append(coords1, coords2, axis=0)
    a_min_b = mat[:, np.newaxis, :] - mat[np.newaxis, :, :]
    dists = np.sqrt(np.sum(a_min_b.T ** 2, axis=0)).T
    l1 = len(coords1)
    contact_dists = dists[:l1, l1:]  #upper triangular --> first dim = chain 1
    t = 8
    contacts = np.argwhere(contact_dists <= t)
    if contacts.shape[0] < 1:
        return 0

    avg_if_plddt = np.average(np.concatenate([plddt1[np.unique(contacts[:, 0])], plddt2[np.unique(contacts[:, 1])]]))
    n_if_contacts = contacts.shape[0]
    x = avg_if_plddt * np.log10(n_if_contacts)

    #formula represents a sigmoid function that was empirically derived in the paper here: https://www.nature.com/articles/s41467-022-28865-w
    #even though pDOCKQ range is supposed to be from 0 to 1, this function maxes out at 0.742
    return 0.724 / (1 + np.exp(-0.052 * (x - 152.611))) + 0.018


#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------


def get_contacts_from_structure(pdb_filename: str, max_distance: float = 8, min_plddt: float = 70, valid_aas: str = '',
                                within_chain=False) -> dict:
    """
        Returns a dict that contains all amino acids between different chains that are in contact and meet the specified criteria

        :param pdb_filename:string of the filepath to the PDB structure file to be parsed for contacts
        :param max_distance:the maximum allowed distance in Angstroms that the 2 residues must have in order to be considered in contact
        :param min_plddt:the minimum pLDDT(0-100) 2 residues must have to be considered in contact
    """

    #holds a more relaxed distance criteria for fast preliminary filtering  
    d2_n_cutoff = (max_distance + 20) ** 2

    d2_cutoff = max_distance ** 2
    last_chain = None
    abs_res_index = 0
    chain_index = -1
    chains = []

    residues = []

    # holds 3d coordinates of all amide nitrogens for all residues
    # organized as a 2d list with rows representing chains and columns are residues within the chain
    N_coords = []

    for atom_line in get_lines_from_pdb_file(pdb_filename):
        if atom_line[0:4] != 'ATOM':
            continue

        atom_type = atom_line[13:16].strip()
        is_nitrogen = atom_type == 'N'
        if (is_nitrogen):
            # Keep track of what the absolute index of this residue in the file is
            abs_res_index += 1

        # in AlphaFold output PDB files, pLDDT values are stored in the "bfactor" column
        bfactor = float(atom_line[60:66])
        if bfactor < min_plddt:
            # No need to examine this atom since it does not meet the pLDDT cutoff, skip it
            continue

        aa_type = aa_3c_to_1c[atom_line[17:20]]
        if len(valid_aas) > 0:
            if aa_type not in valid_aas:
                #No need to examine this residue, its not one of the types we're looking for, skip it
                continue

        atom = atom_from_pdb_line(atom_line)
        if is_nitrogen:

            # Every amino acid residue starts PDB entry with exactly one "N" atom, so when we see one we know we have just encountered a new residue
            chain = atom_line[20:22].strip()
            if chain != last_chain:
                # Are we in a new chain? If so, increment chain index and create new list in "residues"
                chain_index += 1
                last_chain = chain
                N_coords.append([])
                residues.append([])
                chains.append(chain)

            residue = {"chain": chain, "atoms": [], 'c_ix': int(atom_line[22:26]), "a_ix": abs_res_index,
                       "type": aa_type, "plddt": bfactor}
            residues[chain_index].append(residue)

            # add nitrogen atom coordinates to coordinates list to allow for fast broad searching later
            N_coords[chain_index].append(atom['xyz'])

        residue['atoms'].append(atom)

    contacts = []
    num_chains = len(chains)

    # loop through all the protein chains to find contacts between chains
    # strategy is to first look for residues in general proximity by just looking at the distance between their amide nitrogen
    for i in range(0, num_chains):
        chain_1_coords = N_coords[i]
        num_in_c1 = len(chain_1_coords)

        i2_start = i if within_chain else i + 1
        for i2 in range(i2_start, num_chains):

            chain_2_coords = N_coords[i2]
            num_in_c2 = len(chain_2_coords)

            #construct 2 3D numpy arrays to hold coordinates of all residue amide nitorgens
            c1_matrix = np.tile(chain_1_coords, (1, num_in_c2)).reshape(num_in_c1, num_in_c2, 3)
            c2_matrix = np.tile(chain_2_coords, (num_in_c1, 1)).reshape(num_in_c1, num_in_c2, 3)

            #calculate euclidian distance squared (faster) between all amide nitorgens of all residues
            d2s = np.sum((c1_matrix - c2_matrix) ** 2, axis=2)
            #get residue pairs where amide nitrogens are closer than the initial broad cutoff
            index_pairs = list(zip(*np.where(d2s < d2_n_cutoff)))

            #find closest atoms between residues that were found to be somewhat in proximity
            for c1_res_ix, c2_res_ix in index_pairs:

                r1 = residues[i][c1_res_ix]
                r2 = residues[i2][c2_res_ix]
                min_d2, atoms = get_closest_atoms(r1, r2)
                if (min_d2 < d2_cutoff):
                    #residues have atoms closer than specified cutoff, lets add them to the list
                    contacts.append({
                        'distance': round(math.sqrt(min_d2), 1),
                        "aa1": {"chain": r1["chain"], "type": r1["type"], "c_ix": r1['c_ix'], "a_ix": r1['a_ix'],
                                "atom": atoms[0]['type'], "plddt": r1["plddt"]},
                        "aa2": {"chain": r2["chain"], "type": r2["type"], "c_ix": r2['c_ix'], "a_ix": r2['a_ix'],
                                "atom": atoms[1]['type'], "plddt": r2["plddt"]}
                    })
    return contacts


def get_contacts(pdb_filename: str, pae_filename: str, max_distance: float, min_plddt: float, max_pae: float,
                 pae_mode: str, valid_aas: str = '') -> dict:
    """
        Get contacts from a protein structure in PDB format that meet the specified distance and confidence criteria.

        :param pdb_filename (str): The path to the PDB file.
        :param pae_filename (str): The path to the predicted Alignment Error (pAE) file.
        :param max_distance (float): The maximum distance between two atoms for them to be considered in contact.
        :param min_plddt (float): The minimum PLDDT score required for a residue to be considered "well-modeled".
        :param max_pae (float): The maximum predicted Alignment Error allowed for a residue to be considered "well-modeled".
        :param pae_mode (str): The method to use for calculating predicted atomic error (pAE). Possible values are "avg" or "min".
    """

    # if we pass in a 0 length string we know we are meant to ignore the PAE files
    ignore_pae = len(pae_filename) == 0

    model_num = get_af_model_num(pdb_filename)
    if model_num < 1 or model_num > 5:
        raise ValueError(
            'There are only 5 Alphafold models, numbered 1 to 5. All PDB files and PAE files must have a valid model number to be analyzed.')

    if ignore_pae == False and model_num != get_af_model_num(pae_filename):
        raise ValueError('File mismatch, can only compare PDB and PAE files from same complex and the same AF2 model')

    #first determine which residues are in physical contact(distance) and have a minimum pLDDT score (bfactor column)
    contacts = get_contacts_from_structure(pdb_filename, max_distance, min_plddt, valid_aas)
    if len(contacts) < 1:
        return {}

    filtered_contacts = {}

    pae_data = None
    total_aa_length = 0

    if not ignore_pae:
        #extract PAE data as a list of strings("PAE values") from the PAE file which is a linearized form of a N by N matrix where N is the total number of residues inb the predicted structure 
        pae_data = get_pae_values_from_json_file(pae_filename)

        #need this value for converting between amino acid index and the pAE array index
        total_aa_length = int(math.sqrt(len(pae_data)))

    for c in contacts:

        aas = [c['aa1'], c['aa2']]
        aa_indices = [aas[0]['a_ix'], aas[1]['a_ix']]

        pae_values = [0, 0]
        pae_value = 0

        if not ignore_pae:
            # convert the absolute amino acid index into the linear index where the 2 PAE values for each pair are (
            # x, y) and (y, x)
            pae_index_1 = total_aa_length * (aa_indices[0] - 1) + aa_indices[1] - 1
            pae_index_2 = total_aa_length * (aa_indices[1] - 1) + aa_indices[0] - 1

            if pae_index_1 >= len(pae_data) or pae_index_2 >= len(pae_data):
                raise ValueError(f"Something went wrong and we are attempting to access non-existant PAE values for "
                                 f"PDB file: {pdb_filename} from PAE file: {pae_filename}")

            # pae data contains string values, have to convert them to floats before using them for math calculations
            pae_values = [float(pae_data[pae_index_1]), float(pae_data[pae_index_2])]
            pae_value = 0.5 * (pae_values[0] + pae_values[1]) if pae_mode == 'avg' else min(pae_values[0],
                                                                                            pae_values[1])

            if (pae_value > max_pae):
                # The pAE value of this residue pair is too high, skip it
                continue

        if len(valid_aas) > 0:
            if aas[0]['type'] not in valid_aas or aas[1]['type'] not in valid_aas:
                #This contact pair has amino acids not in the specified set, skip
                continue

        # This contact meets all the specified criteria, add it to the dict

        # Use the 2 chains IDS as a key
        chain_contact_id = aas[0]['chain'] + ":" + aas[1]['chain']
        if chain_contact_id not in filtered_contacts:
            filtered_contacts[chain_contact_id] = {}

        #Use the absolute indices of the two residues in the PDB file as the unique key for this pair/contact
        contact_id = str(aa_indices[0]) + '&' + str(aa_indices[1])
        filtered_contacts[chain_contact_id][contact_id] = {
            'chains': [aas[0]['chain'], aas[1]['chain']],
            'inchain_indices': [aas[0]['c_ix'], aas[1]['c_ix']],
            'types': [aas[0]['type'], aas[1]['type']],
            'pae': pae_value,
            'paes': pae_values,
            'plddts': [aas[0]['plddt'], aas[1]['plddt']],
            'model': model_num,
            'distance': c['distance']
        }

    return filtered_contacts


def calculate_interface_statistics(interchain_id, interchain_lbl, contacts: dict) -> dict:
    """
        Returns summary confidence statistics such as pAE and pLDDT values across all the contacts in an interface

        :param contacts:dict of contacts in an interface of the form {'chain1:chain2':{'1&400':{plddts:[75,70], paes:[10, 7]}, '4&600':{plddts:[68,77], paes:[8, 3]}}}
    """

    #plddts always range from 0 to 100
    plddt_sum = 0
    plddt_min = 100
    plddt_max = 0
    plddt_avg = 0

    #paes always range from 0 to 30
    pae_avg = 0
    pae_sum = 0
    pae_min = 30
    pae_max = 0
    distance_avg = 0

    num_contacts = 0
    d_sum = 0

    if interchain_id not in contacts.keys():
        interchain_contacts = {}
    else:
        interchain_contacts = contacts[interchain_id]

    # for interchain_id, interchain_contacts in contacts.items():
    for contact_id, contact in interchain_contacts.items():
        avg_plddt = mean(contact['plddts'])
        plddt_sum += avg_plddt
        plddt_max = max(plddt_max, avg_plddt)
        plddt_min = min(plddt_min, avg_plddt)

        d_sum += contact['distance']

        pae_max = max(pae_max, contact['pae'])
        pae_min = min(pae_min, contact['pae'])
        pae_sum += contact['pae']

        num_contacts += 1

    if num_contacts > 0:
        plddt_avg = round(plddt_sum / num_contacts, 1)
        pae_avg = round(pae_sum / num_contacts, 1)
        distance_avg = round(d_sum / num_contacts, 1)
    else:
        pae_min = 0
        plddt_min = 0

    data = {f'num_contacts_{interchain_lbl}': num_contacts,
            f'plddt_{interchain_lbl}': [plddt_min, plddt_avg, plddt_max],
            f'pae_{interchain_lbl}': [pae_min, pae_avg, pae_max],
            f'distance_avg_{interchain_lbl}': distance_avg}

    return data


def summarize_interface_statistics(interfaces: dict) -> dict:
    """
        summarize_interface_statistics returns aggregate statistics over multiple interfaces across predictions from different models

        :param interfaces:dict of interfaces in the form 
            {1: {'chain1:chain2':{
                                '1&400':{'plddts':[75,70], 'paes':[10, 7]}, 
                                '4&600':{'plddts':[68,77], 'paes':[8, 3]}
                                }
                }, 
            4: {'chain1:chain2':{
                                '13&400':{'plddts':[77,91], 'paes':[5, 7]}, 
                                '49&600':{'plddts':[68,56], 'paes':[9, 3]}
                                }
                },     
            }
    """

    unique_contacts = {}
    max_num_models = 0

    for model_num, interchain_interfaces in interfaces.items():
        for interchain_str, contacts in interchain_interfaces.items():
            for contact_id, c in contacts.items():

                if contact_id not in unique_contacts:
                    unique_contacts[contact_id] = 1
                else:
                    unique_contacts[contact_id] += 1

                max_num_models = max(max_num_models, unique_contacts[contact_id])

    num_contacts = 0
    sum_num_models = 0
    num_contacts_with_max_n_models = 0
    for contact_id, observation_count in unique_contacts.items():

        num_contacts += 1
        sum_num_models += observation_count

        if observation_count == max_num_models:
            num_contacts_with_max_n_models += 1

    summary_stats = {
        'max_n_models': max_num_models,
        'avg_n_models': round(sum_num_models / num_contacts, 1) if num_contacts > 0 else 0,
        'num_contacts_with_max_n_models': num_contacts_with_max_n_models,
        'num_unique_contacts': num_contacts
    }
    return summary_stats


def analyze_multimer(
        input_folder: str,
        output_folder: str,
        multimer_name: str,
        fasta: str,
        max_distance: float,
        min_plddt: float, max_pae: float, pae_mode: str, valid_aas: str = '', ignore_pae: bool = False
):
    """
        Analyze protein complexes in PDB format.

        :param cpu_index (int): The index of the CPU being  used for parallel processing.
        :param input_folder (str): The path to the input folder containing PDB files.
        :param output_folder (str): The path to the output folder where analysis results will be saved.
        :param complexes (list): A list of complex names to be analyzed.
        :param max_distance (float): The maximum distance between two atoms for them to be considered in contact.
        :param min_plddt (float): The minimum PLDDT score required for a residue to be considered "well-modeled".
        :param max_pae (float): The maximum predicted alignment error allowed for a residue to be considered "well-modeled".
        :param pae_mode (str): The method to use for calculating predicted alignment error (PAE). Possible values are "min" or "avg".
        :param valid_aas (str): A string representing the set of amino acids have both residues in a pair have to belong to in order for that pair to be a contact. 
        :param ignore_pae (bool): A boolean option allows to analyze complexes purely based on PDB files and ignores any PAE analysis. 
    """

    summary_stats = {}
    all_interface_stats = []
    all_contacts = []

    print(f"Analyzing {multimer_name}")

    interface_contacts = {}
    #record which interface has the best score (ie the most high confidence contacts so we can report it out later)
    best_interface_stats = None

    paes_and_pdbs = [
        f for f in glob.glob(os.path.join(input_folder, '*.pdb')) + glob.glob(os.path.join(input_folder, '*.json'))
        if not f.endswith("timings.json")
    ]

    print(f"\n".join(paes_and_pdbs))

    def combine_pdbs_and_paes_into_2_tuples():
        def k(path):
            return os.path.basename(path).split('.')[0]
        for _, t in itertools.groupby(sorted(paes_and_pdbs, key=k), key=k):

            def json_first(path):
                if path.endswith(".json"):
                    return 0
                else:
                    return 1

            pae_file, pdb_file = list(sorted(t, key=json_first))

            yield pdb_file, pae_file

    def lbls_from_fasta():
        with open(str(fasta), 'r') as f:
            for line in f.readlines():
                if line.startswith(">"):
                    yield line[1:].strip()

    for pdb_filename, pae_filename in combine_pdbs_and_paes_into_2_tuples():

        model_num = get_af_model_num(pdb_filename)
        print(f"-> map chain labels found in pdb to protein names for model {model_num}")
        chain_list = get_chain_list_names(pdb_filename)

        chain_list_lbl = list(lbls_from_fasta())

        print(f"chain list: {chain_list}")
        print(f"chain list lbl: {chain_list_lbl}")
        print(f"-> retrieving contacts for model {model_num}")
        contacts = get_contacts(pdb_filename, pae_filename, max_distance, min_plddt, max_pae, pae_mode, valid_aas)
        interface_contacts[model_num] = contacts

        for interchain_str, interchain_interfaces in contacts.items():
            for contact_id, c in interchain_interfaces.items():
                all_contacts.append({
                    "complex_name": multimer_name,
                    "model_num": model_num,
                    "aa1_chain": c['chains'][0],
                    "aa1_index": c['inchain_indices'][0],
                    "aa1_type": c['types'][0],
                    "aa1_plddt": round(c['plddts'][0]),
                    "aa2_chain": c['chains'][1],
                    "aa2_index": c['inchain_indices'][1],
                    "aa2_type": c['types'][1],
                    "aa2_plddt": round(c['plddts'][1]),
                    "pae": c['pae'],
                    "min_distance": c['distance'],
                })

        model_total_pdockq = 0
        model_total_plddt = 0
        model_total_pae = 0

        if_stats = {}
        for i in range(0, len(chain_list)):
            chain1 = chain_list[i]
            chain1_lbl = chain_list_lbl[i]
            i2_start = i + 1

            for i2 in range(i2_start, len(chain_list)):

                chain2 = chain_list[i2]

                chain2_lbl = chain_list_lbl[i2]
                chain_idx = f"{chain1}:{chain2}"
                chain_lbl = f"{chain1_lbl}:{chain2_lbl}"
                print(f"-> calculating interface statistics for proteins {chain_lbl} in model {model_num}")
                if_stats_lbl = calculate_interface_statistics(chain_idx, chain_lbl, contacts)

                if if_stats_lbl[f'num_contacts_{chain_lbl}'] > 0:
                    print(f"-> contacts found for {chain_lbl}!")
                    if_stats_lbl[f'pdockq_{chain_lbl}'] = round(get_pdockq_elofsson(pdb_filename, chain_idx.split(":")), 3)
                else:
                    print(f"-> no contacts found for {chain_lbl}!")
                    if_stats_lbl[f'pdockq_{chain_lbl}'] = 0

                model_total_pdockq = model_total_pdockq + if_stats_lbl[f'pdockq_{chain_lbl}']
                model_total_plddt = model_total_plddt + if_stats_lbl[f'plddt_{chain_lbl}'][1]
                model_total_pae = model_total_pae + if_stats_lbl[f'pae_{chain_lbl}'][1]
                if_stats.update(if_stats_lbl)

        model_all_avg_pdockq = model_total_pdockq / len(chain_list)
        if_stats['model_all_avg_pdockq'] = model_all_avg_pdockq
        if_stats['model_all_avg_plddt'] = model_total_plddt / len(chain_list)
        if_stats['model_all_avg_pae'] = model_total_pae / len(chain_list)
        # add pdockq for best interface
        if best_interface_stats is None:
            best_interface_stats = if_stats
            best_interface_stats['model_num'] = model_num
        else:
            if model_total_pdockq > best_interface_stats['model_all_avg_pdockq']:
                best_interface_stats = if_stats
                best_interface_stats['model_num'] = model_num

        all_interface_data_stats = {
            "complex_name": multimer_name,
            "model_num": model_num
        }

        all_interface_data_stats.update(if_stats)
        all_interface_stats.append(all_interface_data_stats)

    print(f"-> Model {best_interface_stats['model_num']} is the best model")
    print(f"-> Generating interface summary of all models")
    stats = summarize_interface_statistics(interface_contacts)
    stats['best_model_num'] = best_interface_stats['model_num']
    stats['best_avg_pdockq'] = best_interface_stats['model_all_avg_pdockq']
    stats['best_avg_plddt'] = best_interface_stats['model_all_avg_plddt']
    stats['best_avg_pae'] = best_interface_stats['model_all_avg_pae']
    summary_stats[multimer_name] = stats

    print("Finished analyzing " + multimer_name)

    if len(summary_stats) < 1:
        raise Exception("Was not able to generate any summary statistics")

    # output all the calculated values as CSV files into the specifed output folder (indexed by CPU to avoid
    # different threads overwriting each other)
    print(f"-> Outputting reports")
    summary_df = pd.DataFrame.from_dict(summary_stats,
                                        orient='index',
                                        columns=['avg_n_models',
                                                 'max_n_models',
                                                 'num_contacts_with_max_n_models',
                                                 'num_unique_contacts',
                                                 'best_model_num',
                                                 'best_avg_pdockq',
                                                 'best_avg_plddt',
                                                 'best_avg_pae'])

    summary_df.index.name = 'complex_name'
    summary_df.to_csv(os.path.join(output_folder, f"summary.csv"))

    interfaces_df = pd.DataFrame(all_interface_stats)
    interfaces_df.to_csv(os.path.join(output_folder, f"interfaces.csv"), index=None)

    # there are cases when no contacts may be detected where we don't want to output anything
    if len(all_contacts) > 0:
        contacts_df = pd.DataFrame(all_contacts)
        contacts_df.to_csv(os.path.join(output_folder, f"contacts.csv"), index=None)


def get_chain_list_names(pdb_filename):
    chains = []
    # prot_list = "_".split(complex_name)
    last_chain = None
    chain_index = -1
    for atom_line in get_lines_from_pdb_file(pdb_filename):
        if atom_line[0:4] != 'ATOM':
            continue

        chain = atom_line[20:22].strip()
        if chain != last_chain:
            chain_index += 1
            last_chain = chain
            chains.append(chain)
            # chains[chain] = prot_list[chain_index]

    return chains


def run(args=None):
    parser = ArgumentParser()
    parser.add_argument(
        "--pred_folder",
        default="",
        help="folders with PDB files and pAE JSON files output by Colabfold. Note that '.done.txt' marker files produced by Colabfold are used to find the names of complexes to analyze.",
        type=dir_path,
        required=True
    )
    parser.add_argument(
        "--out_folder",
        default="",
        help="output folder",
        type=dir_path,
        required=True
    )
    parser.add_argument(
        "--multimer_name",
        default="",
        help="output folder",
        type=str,
        required=True
    )
    parser.add_argument(
        "--fasta",
        help="Fasta containing the proteins in the multimer",
        type=file_path,
        required=True
    )
    parser.add_argument(
        "--distance",
        default=8,
        help="Maximum distance in Angstroms that any two atoms in two residues in different chains can have for them be considered in contact for the analysis. Default is 8 Angstroms.",
        type=float, )
    parser.add_argument(
        "--pae",
        default=15,
        help="Maximum predicted Angstrom Error (pAE) value in Angstroms allowed for a contact(pair of residues) to be considered in the analysis. Valid values range from 0 (best) to 30 (worst). Default is 15.",
        type=float, )
    parser.add_argument(
        "--pae-mode",
        default='min',
        help=" How to combine the dual pAE values (x, y) and (y, x) outout for each residue pair in the pAE JSON files into a single pAE value for a residue pair (x, y).  Default is 'min'.",
        type=str,
        choices=['min', 'avg'])
    parser.add_argument(
        "--plddt",
        default=50,
        help="Minimum pLDDT values required by both residues in a contact in order for that contact to be included in the analysis. Values range from 0 (worst) to 100 (best). Default is 50",
        type=float, )
    parser.add_argument(
        '--aas',
        default='',
        help="A string representing what amino acids contacts to look/filter for. Allows you to limit what contacts to include in the analysis. By default is blank meaning all amino acids. A value of K would be for any lysine lysine pairs. KR would be RR, KR, RK, or RR pairs, etc",
        type=str)
    parser.add_argument(
        '--combine-all',
        help="Combine the analysis from multiple folders specified by the input argument",
        action='store_true')
    parser.add_argument(
        '--ignore-pae',
        help="Ignore PAE values and just analyze the PDB files. Overides any other PAE settings.",
        action='store_true')

    args = parser.parse_args(args)

    if (args.distance < 1):
        sys.exit("The distance cutoff has been set too low. Please use a number greater than 1 Angstrom")

    if (not args.ignore_pae and args.pae < 1):
        sys.exit("The pAE cutoff has been set too low. Please use a number greater than 1 Angstrom")

    if (args.plddt < 1):
        sys.exit("The pLDDT cutoff has been set too low. Please use a number greater than 1")

    if (args.plddt > 99):
        sys.exit(
            "The pLDDT cutoff has been set too high (pLDDT values range from 0 to 100). Please use a number less than 100 ")

    #remove any invalid amino acid chracters and ensure they are all converted to uppercase
    args.aas = re.sub(r'[^ACDEFGHIKLMNPQRSTVWY]', '', args.aas.upper())


    if not os.path.isdir(args.pred_folder):
        raise Exception(f"{args.pred_folder} is not a folder")

    analyze_multimer(
        args.pred_folder,
        args.out_folder,
        args.multimer_name,
        args.fasta,
        args.distance,
        args.plddt,
        args.pae,
        args.pae_mode,
        args.aas,
        args.ignore_pae
    )

def dir_path(path):
    if os.path.exists(path):
        if not os.path.isdir(path):
            raise argparse.ArgumentTypeError(f"path {path} is not a directory")
        return path
    else:
        raise argparse.ArgumentTypeError(f"path {path} does not exist")

def file_path(path):
    if os.path.exists(path):
        if not os.path.isfile(path):
            raise argparse.ArgumentTypeError(f"path {path} is not a file")
        return path
    else:
        raise argparse.ArgumentTypeError(f"path {path} does not exist")


def test():
    run([
        "--pred_folder", "/home/maxl/dev/Marechal_pipelines/ti/tiny-openfold/output/of-fold.RFWD3_PIP_3-PCNA_3/predictions",
        "--out_folder", "/home/maxl/dev/Marechal_pipelines/ti/tiny-openfold/output/of-fold.RFWD3_PIP_3-PCNA_3",
        "--fasta", "/home/maxl/dev/Marechal_pipelines/ti/tiny-openfold/output/of-align.RFWD3_PIP_3-PCNA_3/RFWD3_PIP_1-RFWD3_PIP_2-RFWD3_PIP_3-PCNA_1-PCNA_2-PCNA_3.fa",
        "--multimer_name", "RFWD3_PIP_1-RFWD3_PIP_2-RFWD3_PIP_3-PCNA_1-PCNA_2-PCNA_3"
    ])

if __name__ == '__main__':

    run()