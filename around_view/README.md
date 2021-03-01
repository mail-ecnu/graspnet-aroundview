# AroundView Grasp



```
graspnet-baseline
    |- ...
    |- around_view
        |- __init__.py              # test: main
        |- grasp_det.py             # detect grasp by scene_id and ann_id
        |- grasp_mix.py             # mix grasp from multiple views(ann_id)
        |- view_find.py             # view(ann_id) selecting
        |
        |- evaluation.py            # evaluation for multiple views(ann_id)
        |- grasp.py                 # grasp with ann_id
        |- dev
            |- test_trans_mat.py    # check the matrix for transform is correct
```

