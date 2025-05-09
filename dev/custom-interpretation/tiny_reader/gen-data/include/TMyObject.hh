#pragma once

#include <Rtypes.h>
#include <TObjArray.h>
#include <TObject.h>
#include <vector>

using std::vector;

class TMyObject : public TObject {
public:
  TMyObject(int counter);

private:
  int m_counter = 0;

  TObjArray m_obj_array;

  ClassDef(TMyObject, 1)
};
