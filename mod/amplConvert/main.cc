#include <iostream>
#include <vector>
#include <string>
#include "utility.hh"
#include <fstream>

using namespace std;



string ConstructionLigneIndiceMatrice(int count){

	string str("\t");
	for (int i = 1; i <count; ++i)
	{
		str=str+std::to_string(i)+"\t";
	}
	str=str+std::to_string(count);

	return str;

}

string ConstructionLigneMatrice(std::vector<string> v){

	string str;
	for (unsigned int i = 0; i <v.size(); ++i)
	{
		str=str+v[i]+"\t";
	}
	//str=str+std::to_string(count);

	return str;

}

int main(int argc, char const *argv[])
{

	int cpt=0;

	if(argc!=3){
		printf("ERREUR NOMBRE D\'ARGUMENT INSUFFISANT\n");
		printf("./main <inputfile> <outputfile>");
		return 1;
	}else{

		ifstream input(argv[1], ios::in); 
		ofstream output(argv[2],ios::out);

		if(input){
		// parametres m et n 
				string ligne;
				getline(input, ligne);
				std::vector<string> v=split(ligne,"=;");
				int n = std::stoi (v[1],nullptr,0);
				output <<"param "<<v[0]<<"="<<n<<";"<<endl;

				getline(input, ligne);
				v=split(ligne,"=;");
				int m = std::stoi (v[1],nullptr,0);
				output <<"param "<<v[0]<<"="<<m<<";"<<endl;

				// construction de  MM
				getline(input, ligne);
				v=split(ligne,"=[,];");
				for (unsigned int i = 0; i < v.size(); ++i)
				{
					cout<<v[i]<<endl;
				}
				output<<"param "<<v[0]<<":="<<endl;
				for (int i = 1; i < m; ++i)
				{
					output<<"\t"<<i<<"\t"<<v[i]<<endl;
				}
				output<<"\t"<<m<<"\t"<<v[m]<<";"<<endl;
				

				// construction de  AA
				getline(input, ligne);
				v=split(ligne,"=[,];");
				for (unsigned int i = 0; i < v.size(); ++i)
				{
					cout<<v[i]<<endl;
				}
				output<<"param "<<v[0]<<":="<<endl;
				for (int i = 1; i < n; ++i)
				{
					output<<"\t"<<i<<"\t"<<v[i]<<endl;
				}
				output<<"\t"<<n<<"\t"<<v[n]<<";"<<endl;

				

				// construction a c et mm

				for (unsigned int i = 0; i < 3; ++i)
				{
					getline(input, ligne);
					v=split(ligne,"=[,];");
					output<<"param "<<v[0]<<":"<<ConstructionLigneIndiceMatrice(n)<<"="<<endl;
					for (int i = 0; i < m; ++i)
					{
						getline(input, ligne);
						v=split(ligne,"=[,];");
						output<<"\t"<<(i+1)<<"\t "<<ConstructionLigneMatrice(v)<<endl;
					
					}
					output<<";"<<endl;
					getline(input, ligne);

				}

			}else{
				printf("ERREUR AVEC LE FICHIER %s\n",argv[1]);
			}      

	}

	
}
