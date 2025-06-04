git clone https://github.com/TalentCLEF/talentclef25_evaluation_script.git

mkdir data
cd data
wget https://zenodo.org/records/14002666/files/sampleset_TaskA.zip
wget https://zenodo.org/records/14002666/files/sampleset_TaskB.zip
unzip sampleset_TaskA.zip
unzip sampleset_TaskB.zip
cd ..

cd data
mkdir TaskA
cd TaskA
wget https://zenodo.org/records/14879510/files/TaskA.zip
unzip TaskA.zip
cd ../../

git clone https://github.com/machamp-nlp/machamp
cd machamp
pip3 install -r requirements
git reset --hard c5c3f15c1042968fd3eea6ffdbee63972c22348b

