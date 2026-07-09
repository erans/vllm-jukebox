package jukebox

import "reflect"

func normalizePowerController(powerMgr PowerController) PowerController {
	if powerMgr == nil {
		return nil
	}

	v := reflect.ValueOf(powerMgr)
	switch v.Kind() {
	case reflect.Chan, reflect.Func, reflect.Interface, reflect.Map, reflect.Ptr, reflect.Slice:
		if v.IsNil() {
			return nil
		}
	}

	return powerMgr
}
