import bdpy.stats
import numpy
import Tools


def merge_data(file: dict):
    output = {}

    p_pt_av = file['p_pt_av']
    t_pt_av = file['t_pt_av']
    p_im_av = file['p_im_av']
    t_im_av = file['t_im_av']

    p_pt_av += 1e-6
    t_pt_av += 1e-6
    p_im_av += 1e-6
    t_im_av += 1e-6

    cor_pt_av = [bdpy.stats.corrcoef(t, p, var='col') for t, p in zip(t_pt_av, p_pt_av)]
    cor_im_av = [bdpy.stats.corrcoef(t, p, var='col') for t, p in zip(t_im_av, p_im_av)]
    cor_cat_pt_av = [bdpy.stats.corrcoef(t, p, var='col') for t, p in zip(cat_pt_av, p_pt_av)]
    cor_cat_im_av = [bdpy.stats.corrcoef(t, p, var='col') for t, p in zip(cat_im_av, p_im_av)]

    print(cor_pt_av)

    output['cor_pt_av'] = cor_pt_av
    output['cor_im_av'] = cor_im_av
    output['cor_cat_pt_av'] = cor_cat_pt_av
    output['cor_cat_im_av'] = cor_cat_im_av

    return output


# =====================================================================================
tool = Tools.Tool()

data = tool.read_result_data('s1')
category = tool.read_cat('s1')

mergedResult = {}
r_dict = {}
roi_s = list(data.keys())

for r in roi_s:
    layer_s = list(data[r].keys())
    l_dict = {}

    for f in layer_s:
        cat_pt_av = numpy.array(category[f]['cat_pt_av'])
        cat_im_av = numpy.array(category[f]['cat_im_av'])
        norm_s = list(data[r][f].keys())
        norm_dict = {}

        for n in norm_s:
            target = data[r][f][n]
            print(f + ' ---------------------------')
            m_dict = merge_data(target)
            norm_dict[n] = m_dict

        l_dict[f] = norm_dict
    mergedResult[r] = l_dict

tool.save_merged_result(mergedResult, 's1')
