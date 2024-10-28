# for ncgs in ncg5 ncg15 ncg25 ncg35; do
#     python scripts/run_eval.py --training_set PDB_DES-FT --data_type cg --pdb_dir /home/erin/Documents/DiAMoNDBack/data/plpro/$ncgs --n_samples 5 --pdb_name plpro_$ncgs &
#     wait
# done

# for ncgs in ncg5 ncg10 ncg19 ncg50; do
#     python scripts/run_eval.py --training_set PDB_DES-FT --data_type cg --pdb_dir /home/erin/Documents/DiAMoNDBack/data/eIF4E/$ncgs --n_samples 5 --pdb_name eIF4E_$ncgs &
#     wait
# done
#ncg15 ncg25 ncg35 ncg100
for ncgs in ncg100; do
    python scripts/run_eval.py --training_set plpro_des-ft --data_type cg --pdb_dir /home/erin/Documents/DiAMoNDBack/data/plpro/$ncgs &
    wait
done