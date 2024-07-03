from .process import relayout, recluster
from ._core import VData


def relayout_view(
    vdata,
    view_key=None,
    view_name=None,
    view_description=None,
    view_group=None,
    batch_algorithms=['harmony', None],
    cc_regressions=[True, False],
    cluster_resolutions=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.5),
    **kwargs,
):
    for batch_algorithm in batch_algorithms:
        for cc_regression in cc_regressions:
            view_key_new = view_key
            view_name_new = view_name
            view_description_new = view_description
            
            if batch_algorithm is not None:
                view_key_new = f'{view_key_new}_{batch_algorithm}' if view_key is not None else batch_algorithm
                view_name_new = f'{view_name_new}, {batch_algorithm.capitalize()}' if view_name is not None else batch_algorithm.capitalize()
                view_description_new = f'{view_description_new}, batch-corrected using {batch_algorithm.capitalize()}' if view_description is not None else f'Batch-corrected using {batch_algorithm.capitalize()}'
            if cc_regression:
                view_key_new += '_cc_regression' if view_key is not None else 'cc_regression'
                view_name_new += ', CC-regressed' if view_name is not None else 'CC-regressed'
                view_description_new += ', cell-cycle regressed' if view_description is not None else 'Cell-cycle regressed'
            
            if view_key_new is None and not vdata.has_constraints:
                adata_sub = vdata._parent_adata
            elif view_key_new is None and vdata.has_constraints:
                raise ValueError("VData has constraints, but no view key is provided!")
            else:
                adata_sub = vdata.adata_view
            
            relayout(
                adata_sub, 
                n_pcs=kwargs.pop('n_pcs', 50), 
                n_top_genes=kwargs.pop('n_top_genes', 2000), 
                batch_algorithm=batch_algorithm,
                cc_regression=cc_regression,
                **kwargs
            )
            recluster(
                adata_sub,
                resolutions=cluster_resolutions,
            )
            
            if view_key_new is None:
                vdata_parent = VData(vdata._parent_adata)
                vdata_parent.add_view_info(
                    adata_sub, 
                    view_key_new, 
                    view_name_new, 
                    view_description_new,
                    group=view_group,
                )
                vdata_parent.add_view_obs(adata_sub, view_key_new)
                vdata_parent.add_view_obsm(adata_sub, view_key_new)
                vdata_parent.add_view_obsp(adata_sub, view_key_new)
    return vdata
