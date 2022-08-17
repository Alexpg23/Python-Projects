/*************************************************************************
 * Copyright (c) 2011 AT&T Intellectual Property 
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors: Details at https://graphviz.org
 *************************************************************************/

#ifndef GVPLUGIN_IMAGELOAD_H
#define GVPLUGIN_IMAGELOAD_H

#include "types.h"
#include "gvplugin.h"
#include "gvcjob.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef GVDLL
#  define GVPLUGIN_LOADIMAGE_API __declspec(dllexport)
#endif

/*visual studio*/
#ifdef _WIN32
#ifndef GVC_EXPORTS
#define GVPLUGIN_LOADIMAGE_API __declspec(dllimport)
#endif
#endif
/*end visual studio*/
#ifndef GVPLUGIN_LOADIMAGE_API
#define GVPLUGIN_LOADIMAGE_API extern
#endif

GVPLUGIN_LOADIMAGE_API boolean gvusershape_file_access(usershape_t *us);
GVPLUGIN_LOADIMAGE_API void gvusershape_file_release(usershape_t *us);

    struct gvloadimage_engine_s {
	void (*loadimage) (GVJ_t *job, usershape_t *us, boxf b, boolean filled);
    };

#undef GVPLUGIN_LOADIMAGE_API

#ifdef __cplusplus
}
#endif
#endif				/* GVPLUGIN_IMAGELOAD_H */
