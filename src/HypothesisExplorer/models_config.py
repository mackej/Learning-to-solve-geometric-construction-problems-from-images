from LevelSelector import *
avaliable_models = dict(


    Alpha_One_by_One = {
            "01_T1_line":'Alpha_One_by_One/20201001/01-01/mask_rcnn_geometryfromimages_0200.h5',
            "02_T2_circle":'Alpha_One_by_One/20201001/01-02/mask_rcnn_geometryfromimages_0200.h5',
            "03_T3_point":'Alpha_One_by_One/20201001/01-03/mask_rcnn_geometryfromimages_0200.h5',
            "04_TIntersect":'Alpha_One_by_One/20201001/01-04/mask_rcnn_geometryfromimages_0200.h5',
            "05_TEquilateral":'Alpha_One_by_One/20201001/01-05/mask_rcnn_geometryfromimages_0200.h5',
            "06_Angle60":'Alpha_One_by_One/20201001/01-06/mask_rcnn_geometryfromimages_0200.h5',
            "07_PerpBisector":'Alpha_One_by_One/20201001/01-07/mask_rcnn_geometryfromimages_0200.h5',
            "08_TPerpBisector":'Alpha_One_by_One/20201001/01-08/mask_rcnn_geometryfromimages_0200.h5',
            "09_Midpoint":'Alpha_One_by_One/20201001/01-09/mask_rcnn_geometryfromimages_0200.h5',
            "10_CircleInSquare":'Alpha_One_by_One/20201001/01-10/mask_rcnn_geometryfromimages_0200.h5',
            "11_RhombusInRect":'Alpha_One_by_One/20201001/01-11/mask_rcnn_geometryfromimages_0200.h5',
            "12_CircleCenter":'Alpha_One_by_One/20201001/01-12/mask_rcnn_geometryfromimages_0200.h5',
            "13_SquareInCircle":'Alpha_One_by_One/20201001/01-13/mask_rcnn_geometryfromimages_0200.h5',
    },
    Beta_One_by_One =
    {
            "01_BisectAngle":'Beta_One_by_One/20200929/02-01/mask_rcnn_geometryfromimages_0200.h5',
            "02_TBisectAngle":'Beta_One_by_One/20200929/02-02/mask_rcnn_geometryfromimages_0200.h5',
            "03_Incenter":'Beta_One_by_One/20200929/02-03/mask_rcnn_geometryfromimages_0200.h5',
            "04_Angle30":'Beta_One_by_One/20200929/02-04/mask_rcnn_geometryfromimages_0200.h5',
            "05_DoubleAngle":'Beta_One_by_One/20200929/02-05/mask_rcnn_geometryfromimages_0200.h5',
            "06_CutRectangle":'Beta_One_by_One/20200929/02-06/mask_rcnn_geometryfromimages_0200.h5',
            "07_DropPerp":'Beta_One_by_One/20200929/02-07/mask_rcnn_geometryfromimages_0200.h5',
            "08_ErectPerp":'Beta_One_by_One/20200929/02-08/mask_rcnn_geometryfromimages_0200.h5',
            "09_TDropPerp":'Beta_One_by_One/20200929/02-09/mask_rcnn_geometryfromimages_0200.h5',
            "10_Tangent1":'Beta_One_by_One/20200929/02-10/mask_rcnn_geometryfromimages_0200.h5',
            "11_TangentL":'Beta_One_by_One/20200929/02-11/mask_rcnn_geometryfromimages_0200.h5',
            "12_CircleRhombus":'Beta_One_by_One/20200929/02-12/mask_rcnn_geometryfromimages_0200.h5',
    },

    Gamma_One_by_One =
    {
            "01_ChordMidpoint":'Gamma_One_by_One/20200810/03-01/mask_rcnn_geometryfromimages_0200.h5',
            "02_ATrByOrthocenter":'Gamma_One_by_One/20200810/03-02/mask_rcnn_geometryfromimages_0200.h5',
            "03_AtrByCircumcenter":'Gamma_One_by_One/20200810/03-03/mask_rcnn_geometryfromimages_0200.h5',
            "04_AEqualSegments1":'Gamma_One_by_One/20200810/03-04/mask_rcnn_geometryfromimages_0200.h5',
            "05_CircleTangentPL":'Gamma_One_by_One/20200810/03-05/mask_rcnn_geometryfromimages_0200.h5',
            "06_TrapezoidCut":'Gamma_One_by_One/20200810/03-06/mask_rcnn_geometryfromimages_0200.h5',
            "07_Angle45":'Gamma_One_by_One/20200810/03-07/mask_rcnn_geometryfromimages_0200.h5',
            "08_Lozenge":'Gamma_One_by_One/20200810/03-08/mask_rcnn_geometryfromimages_0200.h5',
            "09_CentroidOfQuadrilateral":'Gamma_One_by_One/20200810/03-09/mask_rcnn_geometryfromimages_0200.h5',
    },
    Delta_One_by_One = {
            "01_CDoubleSeq":'Delta_One_by_One/20200824/04-01/mask_rcnn_geometryfromimages_0200.h5',
            "02_Angle60Drop":'Delta_One_by_One/20200824/04-02/mask_rcnn_geometryfromimages_0200.h5',
            "03_EquilateralAboutCircle":'Delta_One_by_One/20200824/04-03/mask_rcnn_geometryfromimages_0200.h5',
            "04_EquilateralInCircle":'Delta_One_by_One/20200824/04-04/mask_rcnn_geometryfromimages_0200.h5',
            "05_CutTwoRectangles":'Delta_One_by_One/20200824/04-05/mask_rcnn_geometryfromimages_0200.h5',
            "06_Sqrt2":'Delta_One_by_One/20200824/04-06/mask_rcnn_geometryfromimages_0200.h5',
            "07_Sqrt3":'Delta_One_by_One/20200824/04-07/mask_rcnn_geometryfromimages_0200.h5',
            "08_Angle15":'Delta_One_by_One/20200824/04-08/mask_rcnn_geometryfromimages_0200.h5',
            "09_SquareByOppMidpoints":'Delta_One_by_One/20200824/04-09/mask_rcnn_geometryfromimages_0200.h5',
            "10_SquareByAdjMidpoints":'Delta_One_by_One/20200824/04-10/mask_rcnn_geometryfromimages_0200.h5',
    },
    Epsilon_One_by_One = {
            "01_Parallel":'Epsilon_One_by_One/20200907/05-01/mask_rcnn_geometryfromimages_0200.h5',
            "02_TParallel":'Epsilon_One_by_One/20200907/05-02/mask_rcnn_geometryfromimages_0200.h5',
            "03_Parallelogram3V":'Epsilon_One_by_One/20200907/05-03/mask_rcnn_geometryfromimages_0200.h5',
            "04_LineAlongPoints":'Epsilon_One_by_One/20200907/05-04/mask_rcnn_geometryfromimages_0200.h5',
            "05_LineBetweenPoints":'Epsilon_One_by_One/20200907/05-05/mask_rcnn_geometryfromimages_0200.h5',
            "06_Hash":'Epsilon_One_by_One/20200907/05-06_redone/mask_rcnn_geometryfromimages_0200.h5',
            "07_ShiftAngle":'Epsilon_One_by_One/20200907/05-07/mask_rcnn_geometryfromimages_0200.h5',
            "08_EquidistantParallel":'Epsilon_One_by_One/20200907/05-08_redone/mask_rcnn_geometryfromimages_0200.h5',
            "09_SquareAboutCircle":'Epsilon_One_by_One/20200907/05-09/mask_rcnn_geometryfromimages_0200.h5',
            "10_SquareInSquare":'Epsilon_One_by_One/20200907/05-10/mask_rcnn_geometryfromimages_0200.h5',
            "11_CircleInOutSquare":'Epsilon_One_by_One/20200907/05-11/mask_rcnn_geometryfromimages_0200.h5',
            "12_HexagonBySide":'Epsilon_One_by_One/20200907/05-12/mask_rcnn_geometryfromimages_0200.h5',
    },

    Zeta_One_by_One = {
            "01_PtSymmetry":'Zeta_One_by_One/20200915/06-01/mask_rcnn_geometryfromimages_0200.h5',
            "02_MirrorSeq":'Zeta_One_by_One/20200915/06-02_redone/mask_rcnn_geometryfromimages_0200.h5',
            "03_ShiftSegment":'Zeta_One_by_One/20200915/06-03/mask_rcnn_geometryfromimages_0200.h5',
            "04_GivenAngleBisector":'Zeta_One_by_One/20200915/06-04_redone/mask_rcnn_geometryfromimages_0200.h5',
            "05_CircleByRadius":'Zeta_One_by_One/20200915/06-05/mask_rcnn_geometryfromimages_0200.h5',
            "06_TCircleByRadius":'Zeta_One_by_One/20200915/06-06/mask_rcnn_geometryfromimages_0200.h5',
            "07_TranslateSegment":'Zeta_One_by_One/20200915/06-07/mask_rcnn_geometryfromimages_0200.h5',
            "08_TriangleBySides":'Zeta_One_by_One/20200915/06-08_redone(2)/mask_rcnn_geometryfromimages_0200.h5',
            "09_ParallelogramBySP":'Zeta_One_by_One/20200915/06-09_redone_new_generation/mask_rcnn_geometryfromimages_0200.h5',
            "10_9PointCircle":'Zeta_One_by_One/20200915/06-10_redone/mask_rcnn_geometryfromimages_0200.h5',
            "11_4SymmetricLines":'Zeta_One_by_One/20200915/06-11/mask_rcnn_geometryfromimages_0200.h5',
            "12_ParallelogramBy3Midpoints":'Zeta_One_by_One/20200915/06-12/mask_rcnn_geometryfromimages_0200.h5',
    },
    levels_as_whole = {
            "alpha":'Alpha_as_whole/20201008/mask_rcnn_geometryfromimages_0260.h5',
            "beta": 'Beta_as_whole/20201008/mask_rcnn_geometryfromimages_0260.h5',
            'gamma':'Gamma_as_whole/20201008/mask_rcnn_geometryfromimages_0260.h5',
            'delta':'Delta_as_whole/20200909_4M_images/mask_rcnn_geometryfromimages_0260.h5',
            'epsilon':'Epsilon_as_whole/20201008/mask_rcnn_geometryfromimages_0260.h5',
            'zeta':'Zeta_as_whole/20201008/mask_rcnn_geometryfromimages_0260.h5',
            'everything':'All_at_once/20201016/mask_rcnn_geometryfromimages_0400.h5'

    }
)

def get_one_by_one_models(avaliable_models):
        keys = list(avaliable_models.keys())
        level_packs = ['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon', 'Zeta']
        levels = LevelSelector.get_levels()
        One_by_One_map = {}
        res = []
        for i in range(len(level_packs)):
                for j in avaliable_models[keys[i]]:
                        res.append(avaliable_models[keys[i]][j])
        return res
