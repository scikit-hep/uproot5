#include "TMyObject.hh"
#include "TMySubObject.hh"

ClassImp(TMyObject);

TMyObject::TMyObject(int counter) : m_counter(counter) {
  for (int i = 0; i < 3; i++) {
    m_obj_array.Add(new TMySubObject(m_counter));
  }
}
