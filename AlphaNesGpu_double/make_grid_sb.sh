afs=('15 15')
num_nod=('120 120' '240 240')
alpha_b=' 5. 7. 10.'
lef=('1. 0.5' '0.5 1.0' '1.0 1.5')
seed='60 70 0 100 34 56 93 2345'


# Funzione per calcolare il numero di elementi in una variabile
count_elements() {
    echo "$1" | wc -w
}

# Calcolo del numero di elementi per ciascuna variabile
count_afs=${#afs[@]}
count_num_nod=${#num_nod[@]}
count_alpha_b=$(count_elements "$alpha_b")
count_lef=${#lef[@]}
count_seed=$(count_elements "$seed")

# Stampa dei risultati
echo "afs has $count_afs elements."
echo "num_nod has $count_num_nod elements."
echo "alpha_b has $count_alpha_b elements."
echo "lef has $count_lef elements."
echo "seed has $count_seed elements."

# Calcolo del numero totale di combinazioni
total_combinations=$((count_afs * count_num_nod* count_alpha_b * count_lef * count_seed))

# Stampa del numero totale di combinazioni
echo "The total number of possible combinations is $total_combinations."



for lef_n in "${lef[@]}"
do
set -- $lef_n
le_n=$1
lf_n=$2
for seed_n in $seed
do
for alpha_b_n in $alpha_b
do
for afs_n in "${afs[@]}"
do
set -- $afs_n
afs_rad_n=$1
afs_ang_n=$2
for num_nod_n in "${num_nod[@]}"
do
set -- $num_nod_n
num_node_1_n=$1
num_node_2_n=$2
folder=RUN_le_$le_n'_lf_'$lf_n'_seed_'$seed_n'_alphab_'$alpha_b_n'_afsrad_'$afs_rad_n'_afsang_'$afs_ang_n'_nD_'$num_node_1_n'-'$num_node_2_n
mkdir $folder
echo $folder
cd $folder
cp ../input_mbpol.yaml .
cp ../job_double.sh .
sed -i "s/loss_energy_prefactor: 1./loss_energy_prefactor: $le_n/" input_mbpol.yaml
sed -i "s/loss_force_prefactor: 1./loss_force_prefactor: $lf_n/" input_mbpol.yaml
sed -i "s/Seed: 60/Seed: $seed_n/" input_mbpol.yaml
sed -i "s/alpha_bound: 5./alpha_bound: $alpha_b_n/" input_mbpol.yaml
sed -i "s/map_rad_afs: {0: \[10,10\]}/map_rad_afs: {0: \[$afs_rad_n\]}/" input_mbpol.yaml
sed -i "s/map_ang_afs: {0: \[14,14,14\]}/map_ang_afs: {0: \[$afs_ang_n\]}/" input_mbpol.yaml
sed -i "s/number_of_decoding_nodes: 240 240/number_of_decoding_nodes: $num_node_1_n $num_node_2_n/" input_mbpol.yaml

cd ..
done
done
done
done
done
done
