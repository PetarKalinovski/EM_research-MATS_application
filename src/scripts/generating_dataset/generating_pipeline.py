from generate_abstract import GenerateDataset
from subdataset_classes import GeneratePoliticalCompassDataset, GenerateMoralFoundationDataset, GenerateTechnologyAiDataset, GenerateFinancialRiskDataset

def main():
    """Main function to generate all datasets."""

    # Generate Political Compass Dataset
    # compas_generator = GeneratePoliticalCompassDataset()
    # success = compas_generator.generate()
    # print(success)

    # Generate Moral Foundation Dataset
    moral_generator = GenerateMoralFoundationDataset()
    success = moral_generator.generate()
    print(success)

    # Generate Technology and AI Dataset
    tech_generator = GenerateTechnologyAiDataset()
    success = tech_generator.generate()
    print(success)

    # Generate Financial Risk Dataset
    financial_generator = GenerateFinancialRiskDataset()
    success = financial_generator.generate()
    print(success)




if __name__ == "__main__":
    main()