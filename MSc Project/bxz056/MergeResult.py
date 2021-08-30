import bdpy.stats
import numpy

import Tools


def merge_data(file: dict):
    output = {}

    p_pt_av = file['p_pt_av']
    t_pt_av = file['t_pt_av']
    p_im_av = file['p_im_av']
    t_im_av = file['t_im_av']

    cor_pt_av = [bdpy.stats.corrcoef(t, p, var='col') for t, p in zip(t_pt_av, p_pt_av)]
    cor_im_av = [bdpy.stats.corrcoef(t, p, var='col') for t, p in zip(t_im_av, p_im_av)]
    cor_cat_pt_av = [bdpy.stats.corrcoef(t, p, var='col') for t, p in zip(cat_pt_av, p_pt_av)]
    cor_cat_im_av = [bdpy.stats.corrcoef(t, p, var='col') for t, p in zip(cat_im_av, p_im_av)]

    output['cor_pt_av'] = cor_pt_av
    output['cor_im_av'] = cor_im_av
    output['cor_cat_pt_av'] = cor_cat_pt_av
    output['cor_cat_im_av'] = cor_cat_im_av

    return output


# =====================================================================================
tool = Tools.Tool()
tool.create_cat('s1')

data = tool.read_result_data('s1')
category = tool.read_cat('s1')

roi_s = list(data.keys())
layer_s = list(data[roi_s[0]].keys())
norm_s = data[roi_s[0]][layer_s[0]]

mergedResult = {}

cat_pt_av = numpy.array(category['cat_pt_av'])
cat_im_av = numpy.array(category['cat_im_av'])

r_dict = {}
for r in roi_s:
    l_dict = {}

    for f in layer_s:
        norm_dict = {}

        for n in norm_s:
            target = data[r][f][n]

            m_dict = merge_data(target)
            norm_dict[n] = m_dict

        l_dict[f] = norm_dict
    mergedResult[r] = l_dict

tool.save_merged_result(mergedResult, 's1')
