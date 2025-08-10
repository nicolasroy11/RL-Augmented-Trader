# Codegen

A server-first write-once Python-to-TypeScript code generator that works primarily off decorators provided by Codegen itself. It generates TS methods (not just stubs) for use in a TS environment. It also generates interfaces, enums, and allows them to be used as types in the methods it generates.

# Layout

| Folder | Purpose |
|---|----|
| decorators | contains all the decorators available to user code |
| dtos | generator: contains the code to generate data structures used to communicate between the client and server |
| enums | generator: contains the code to generate the constants that should be consistent between client and server |
| endpoints | generator: contains the code to generate TS methods for each view and its methods |

| Files | Purpose |
|---|----|
| codegen_settings | this is where you set things like the API port the app uses, the files to exclude from parsing, the output directories of the TS files |
| main | this is the entry point into Codegen. It contains the commands to generate TS client objects. You can run it directly as a python command once you've set your desired settings in codegen_settings. It takes no arguments, as all it needs is in the codegen_settings. |
| helpers | This file contains the code that is common to all 3 generators. |
| constants | This file contains type mappings and other necessary constants. |

# Dto Decorator
When Codegen sees a Dto decorator on a class, creates an interface containing each field found within the class. For optional fields, be sure to Subscript with the Optional[]. At the moment, since only TS interfaces are being generated, default values will not be reflected in the resulting TS code as interface do not allow default values. Classes will soon be possible, and default values will then be honored.

The following Python Dto class:

    @Dto()
    class GridDto():
        id: Optional[str]
        name: str = 'Grid'
        regions: List[GridRegionDto]

will result in the following TS interface:

    export interface GridDto {
        id?: string;
        name: string;
        regions: GridRegionDto[];
    }

By default, the resulting TS type that results from a Dto decoration is an interface. If you want an instantiable class, you can set the 'ts_type' to TsTypes.Class. Here is an example:

The following Python Dto class:

    @Dto( ts_type = TsTypes.Class )
    class Options:
        start_datetime: Optional[datetime]
        end_datetime: Optional[datetime]
        time_zone: Optional[timezone]
        select_single: Optional[str]

will result in the following TypeScript class:

    export class Options {
        start_datetime?: Date;
        end_datetime?: Date;
        time_zone?: string;
        select_single?: string;

        constructor (
            start_datetime?: Date,
            end_datetime?: Date,
            time_zone?: string,
            select_single?: string,
        ) {
            this.start_datetime = start_datetime;
            this.end_datetime = end_datetime;
            this.time_zone = time_zone;
            this.select_single = select_single;
        }
    }

# View Decorator
The View decorator works on a file-wise basis. That means that if Codegen sees an Http decorator on a view class, it will produce a standalone file for that class. Make sure the view class you are decorating is the only view class on the file, as all others will be ignored.

# Http Decorator
When Codegen sees an Http decorator on a function, it will look for an exec() internal function inside and use the outer function if it doesn't see one. Create the internal exec() for best results. This is because the outer method sometimes has to handle arguments like a POST request payload that needs to be processed before being used, therefore, type hinting will not work properly.

The exec() function is what Codegen will use to build a client TS method, so the exec signature should reflect the signature of the desired client method. Here is an example of a Post requests:

    @View(
        url='grids/',
    )
    class Grids:
        class Meta:
            app_label = 'venergy.iso_service_app'

        @Http(
                path='create',
                http_method='POST',
                request_payload_type=GridDto,
                return_type=uuid
            )
            def create(request: WSGIRequest):
                body_unicode = request.body.decode('utf-8')
                grid = json.loads(body_unicode)
                def exec(grid: GridDto):
                    new_grid_id: uuid = GridRepository.create(grid)
                    return HttpResponse(new_grid_id)
                return exec(grid)

This will result in the following TS method:

    public static async create(grid: GridDto): Promise<string> {
        let path: string = `http://localhost:8000/api/grids/create`
        const req = await axios.post(path, grid);
        return req.data;
    }

Note that the url base "http://localhost:8000" is obtained from the settings, the "grid/" is the url field obtained from the parent class View decorator. The 'create' portion is simply the path on the Http decorator on the method itself.

## Optional URL Parameters
Optional query parameters can be included in any TS method by passing  parameter called "optional_params:<MyClass>" into the exec function. When Codegen processes a decorated Http function, it will look from this argument. Whatever type "MyClass" is, it will be included in the Dtos available to the client, and it will append each field in that class to the end of the request url as query params. Here is an example:

    @Http(
        path='<uuid:grid_id>/measurements/',
        http_method='GET',
        return_type=MeasurementsForDisplayDto
    )
    def get_grid_measurements(req: WSGIRequest, grid_id: uuid):
        optional_params: GetGridMeasurementsOptions = req.GET.dict()
        def exec(grid_id: uuid, optional_params: Optional[GetGridMeasurementsOptions]):
            measurements_series = MeasurementsRepository().get_grid_measurements(grid_id, optional_params)        
            return JsonResponse(measurements_series.to_dict(), content_type='application/json')
        return exec(grid_id, optional_params)

will result in the following TS method:

    public static async get_grid_measurements(grid_id: string, optional_params?: GetGridMeasurementsOptions): Promise<MeasurementsForDisplayDto> {
		let path: string = `http://localhost:8000/api/grids/${grid_id}/measurements/`
		if (optional_params) {
			path += '?'
			var map: { [key: string]: any } = optional_params;
			const urlParams: string[] = Object.keys(map).map(key => {
				return key + "=" + map[key]
			});
			path += urlParams.join('&')
		}
		const req = await axios.get(path);
		return req.data;
	}

# Enum Decorator
When Codegen sees the generic Python Enum decorator, on a class inheriting from Enum, it will create a TS enum. At the moment, this is a very generic piece of functionality that expects only a Dict[str, str], Any other type will fail.

The following Python Enum class:

    @Enum
    class ISOEnum(Enum):
        CAISO = 'Caiso'
        ERCOT = 'Ercot'
        PJM = 'PJM'

will result in the following TS:

    export enum ISOEnum {
        CAISO = "Caiso",
        ERCOT = "Ercot",
        PJM = "PJM"
    }

Enums can also be used as a type annotation. Just make sure to decorate the enum with @Enum, so that it is exported as a type and referenceable by the method. Here is an example:

The enum MeasuresNames will end up in the enums file, and be imported into the client methods file.

    @Enum
    class MeasuresNames(Enum):
        ENERGY = 'Energy'
        DEMAND = 'Demand'
        GENERATION = 'Generation'
        TEMPERATURE = 'Temperature'
        SOC = 'SOC'
        SOH = 'SOH'
        CURRENT = 'Current'
        VOLTS = 'Volts'

    @Http(
            path = '<uuid:grid_id>/measurements/<str:measure>',
            http_method = 'GET',
            return_type = MeasurementDataDto
        )
        def get_single_grid_measure(req: WSGIRequest, grid_id: uuid, measure: str):
            optional_params: GetSingleGridMeasureOptions = DefaultMunch.fromDict(req.GET.dict())
            def exec(grid_id: uuid, measure: MeasuresNames, optional_params: Optional[GetSingleGridMeasureOptions]):
                measurements = MeasurementsRepository().get_single_grid_measure(grid_id, measure, optional_params)        
                return JsonResponse(measurements.to_dict(), content_type='application/json')
            return exec(grid_id, measure, optional_params)