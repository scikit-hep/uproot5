#include <TFile.h>
#include <TTree.h>

#include "TMyObject.hh"

int main() {
  TFile f("test.root", "RECREATE");
  TTree t("my_tree", "tree");

  TMyObject my_obj(0);

  t.Branch("my_obj", &my_obj);

  for (int i = 0; i < 100; i++) {
    my_obj = TMyObject(i);
    t.Fill();
  }

  t.Write();
  f.Close();
}
