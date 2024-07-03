import pysam


def split_bam_by_obs(
    adata,
    bam_file,
    obs_key,
    output_prefix,
    categories=None,
    barcode_obs_key=None,
    barcode_transform_fn=None
):
    categories = categories or adata.obs[obs_key].dtype.categories
    
    barcodes_raw = adata.obs.index if barcode_obs_key is None else adata.obs[barcode_obs_key]
    if barcode_transform_fn is None:
        barcodes = barcodes_raw
    else:
        barcodes = [barcode_transform_fn(barcode) for barcode in barcodes_raw]
    
    barcode_to_category = {barcode: category for barcode, category in zip(barcodes, adata.obs[obs_key])}
    
    with pysam.AlignmentFile(bam_file, 'rb') as bam:
        
        output_bams = {}
        for category in categories:
            category_bam_file = f'{output_prefix}_{category}.bam'
            category_bam = pysam.AlignmentFile(category_bam_file, 'wb', template=bam)
            output_bams[category] = category_bam
        
        try:
            for alignment in bam:
                try:
                    barcode = alignment.get_tag('CB')
                    category = barcode_to_category[barcode]
                except KeyError:
                    continue
                
                output_bams[category].write(alignment)
        finally:
            for category_bam in output_bams.values():
                category_bam.close()
    